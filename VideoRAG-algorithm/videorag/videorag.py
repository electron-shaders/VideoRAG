import os
import sys
import json
import shutil
import asyncio
import multiprocessing
from multiprocessing.managers import SyncManager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Callable, Dict, List, Optional, Type, Union, cast
import tiktoken


from ._llm import (
    LLMConfig,
    openai_config,
    azure_openai_config,
    ollama_config
)
from ._op import (
    chunking_by_video_segments,
    extract_entities,
    get_chunks,
    videorag_query,
    videorag_query_multiple_choice,
)
from ._storage import (
    JsonKVStorage,
    NanoVectorDBStorage,
    NanoVectorDBVideoSegmentStorage,
    NetworkXStorage,
)
from ._workers import run_video_workers
from ._caption_model import load_caption_model, resolve_caption_snapshot
from ._recovery import UnsafeVideoRAGState
from ._utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    limit_async_func_call,
    wrap_embedding_func_with_attrs,
    convert_response_to_json,
    always_get_an_event_loop,
    logger,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
    QueryParam,
)
from ._videoutil import(
    split_video,
    speech_to_text,
    segment_caption,
    merge_segment_information,
    saving_video_segments,
)


@dataclass
class VideoRAG:
    working_dir: str = field(
        default_factory=lambda: f"./videorag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    
    # video
    threads_for_split: int = 10
    video_segment_length: int = 30 # seconds
    rough_num_frames_per_segment: int = 5 # frames
    fine_num_frames_per_segment: int = 15 # frames
    video_output_format: str = "mp4"
    audio_output_format: str = "mp3"
    video_embedding_batch_num: int = 2
    segment_retrieval_top_k: int = 4
    video_embedding_dim: int = 1024
    caption_model_path: Optional[str] = None
    
    # query
    retrieval_topk_chunks: int = 2
    query_better_than_threshold: float = 0.2
    multiple_choice_response_max_attempts: int = 3
    
    # graph mode
    enable_local: bool = True
    enable_naive_rag: bool = True

    # text chunking
    chunk_func: Callable[
        [
            list[list[int]],
            List[str],
            tiktoken.Encoding,
            Optional[int],
        ],
        List[Dict[str, Union[str, int]]],
    ] = chunking_by_video_segments
    chunk_token_size: int = 1200
    # chunk_overlap_token_size: int = 100
    tiktoken_model_name: str = "gpt-4o"

    # entity extraction
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500

    # Change to your LLM provider
    llm: LLMConfig = field(default_factory=openai_config)
    
    # entity extraction
    entity_extraction_func: callable = extract_entities
    
    # storage
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBStorage
    vs_vector_db_storage_cls: Type[BaseVectorStorage] = NanoVectorDBVideoSegmentStorage
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)
    graph_storage_cls: Type[BaseGraphStorage] = NetworkXStorage
    enable_llm_cache: bool = True

    # extension
    always_create_working_dir: bool = True
    addon_params: dict = field(default_factory=dict)
    convert_response_to_json_func: callable = convert_response_to_json

    def load_caption_model(self, debug=False):
        # caption model
        if not debug:
            self.caption_model_path = resolve_caption_snapshot(self.caption_model_path)
            self.caption_model, self.caption_tokenizer = load_caption_model(self.caption_model_path)
        else:
            self.caption_model = None
            self.caption_tokenizer = None
    
    def __post_init__(self):
        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"VideoRAG init with param:\n\n  {_print_config}\n")
        
        if not os.path.exists(self.working_dir) and self.always_create_working_dir:
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        self.video_path_db = self.key_string_value_json_storage_cls(
            namespace="video_path", global_config=asdict(self)
        )
        
        self.video_segments = self.key_string_value_json_storage_cls(
            namespace="video_segments", global_config=asdict(self)
        )

        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks", global_config=asdict(self)
        )

        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache", global_config=asdict(self)
            )
            if self.enable_llm_cache
            else None
        )

        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation", global_config=asdict(self)
        )

        self.embedding_func = limit_async_func_call(self.llm.embedding_func_max_async)(wrap_embedding_func_with_attrs(
                embedding_dim = self.llm.embedding_dim,
                max_token_size = self.llm.embedding_max_token_size,
                model_name = self.llm.embedding_model_name)(self.llm.embedding_func))
        self.entities_vdb = (
            self.vector_db_storage_cls(
                namespace="entities",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
                meta_fields={"entity_name"},
            )
            if self.enable_local
            else None
        )
        self.chunks_vdb = (
            self.vector_db_storage_cls(
                namespace="chunks",
                global_config=asdict(self),
                embedding_func=self.embedding_func,
            )
            if self.enable_naive_rag
            else None
        )
        
        self.video_segment_feature_vdb = (
            self.vs_vector_db_storage_cls(
                namespace="video_segment_feature",
                global_config=asdict(self),
                embedding_func=None, # we code the embedding process inside the insert() function.
            )
        )
        
        self.llm.best_model_func = limit_async_func_call(self.llm.best_model_max_async)(
            partial(self.llm.best_model_func, hashing_kv=self.llm_response_cache)
        )
        self.llm.cheap_model_func = limit_async_func_call(self.llm.cheap_model_max_async)(
            partial(self.llm.cheap_model_func, hashing_kv=self.llm_response_cache)
        )

    def insert_video(self, video_path_list=None):
        loop = always_get_an_event_loop()
        new_video_segments = {}
        for video_path in video_path_list:
            # Step0: check the existence
            video_name = os.path.basename(video_path).split('.')[0]
            if video_name in self.video_segments._data:
                logger.info(f"Find the video named {os.path.basename(video_path)} in storage and skip it.")
                continue
            loop.run_until_complete(self.video_path_db.upsert(
                {video_name: video_path}
            ))
            
            # Step1: split the videos
            segment_index2name, segment_times_info = split_video(
                video_path, 
                self.working_dir, 
                self.video_segment_length,
                self.rough_num_frames_per_segment,
                self.audio_output_format,
            )
            
            # Step2: obtain transcript with whisper
            transcripts = speech_to_text(
                video_name, 
                self.working_dir, 
                segment_index2name,
                self.audio_output_format
            )
            
            # Step3: saving video segments **as well as** obtain caption with vision language model
            ctx = multiprocessing.get_context("spawn")
            # Keep manager IPC off shared TMPDIR filesystems: Unix-socket
            # cleanup can leave busy .__dpc/.nfs entries during shutdown.
            # Loopback + port 0 stays on this node and needs no fixed port.
            with SyncManager(address=("127.0.0.1", 0), ctx=ctx, shutdown_timeout=5.0) as manager:
                captions = manager.dict()
                error_queue = manager.Queue()
                workers = [
                    ctx.Process(
                        name="videorag-save-segments",
                        target=saving_video_segments,
                        args=(
                            video_name,
                            video_path,
                            self.working_dir,
                            segment_index2name,
                            segment_times_info,
                            error_queue,
                            self.video_output_format,
                        ),
                    ),
                    ctx.Process(
                        name="videorag-caption-segments",
                        target=segment_caption,
                        args=(
                            video_name,
                            video_path,
                            segment_index2name,
                            transcripts,
                            segment_times_info,
                            captions,
                            error_queue,
                            self.caption_model_path,
                        ),
                    ),
                ]
                try:
                    run_video_workers(workers, error_queue)
                except Exception as exc:
                    with open('error_log_videorag.txt', 'a', encoding='utf-8') as log_file:
                        log_file.write(f"Video Name:{video_name} Error processing:\n{exc}\n\n")
                    raise

                # Step4: detach results before the manager shuts down. Worker
                # cleanup finishes inside run_video_workers, even on failure.
                segments_information = merge_segment_information(
                    segment_index2name,
                    segment_times_info,
                    transcripts,
                    dict(captions),
                )
            loop.run_until_complete(self.video_segments.upsert(
                {video_name: segments_information}
            ))
            new_video_segments[video_name] = segments_information
            
            # Step5: encode video segment features
            loop.run_until_complete(self.video_segment_feature_vdb.upsert(
                video_name,
                segment_index2name,
                self.video_output_format,
            ))
            
            # Step6: delete the cache file
            video_segment_cache_path = os.path.join(self.working_dir, '_cache', video_name)
            if os.path.exists(video_segment_cache_path):
                shutil.rmtree(video_segment_cache_path)
            
        # Persist video metadata/features together with text and graph indexes.
        # If indexing fails, a restarted process must not treat the video as a
        # completed insertion.
        if new_video_segments:
            loop.run_until_complete(self.ainsert(new_video_segments))

    def query(self, query: str, param: QueryParam = QueryParam()):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param))

    async def aquery(self, query: str, param: QueryParam = QueryParam()):
        if param.mode == "videorag":
            response = await videorag_query(
                query,
                self.entities_vdb,
                self.text_chunks,
                self.chunks_vdb,
                self.video_path_db,
                self.video_segments,
                self.video_segment_feature_vdb,
                self.chunk_entity_relation_graph,
                self.caption_model, 
                self.caption_tokenizer,
                param,
                asdict(self),
            )
        # NOTE: update here
        elif param.mode == "videorag_multiple_choice":
            response = await videorag_query_multiple_choice(
                query,
                self.entities_vdb,
                self.text_chunks,
                self.chunks_vdb,
                self.video_path_db,
                self.video_segments,
                self.video_segment_feature_vdb,
                self.chunk_entity_relation_graph,
                self.caption_model, 
                self.caption_tokenizer,
                param,
                asdict(self),
            )
        else:
            raise ValueError(f"Unknown mode {param.mode}")
        await self._query_done()
        return response

    async def ainsert(self, new_video_segment):
        await self._insert_start()
        try:
            # ---------- chunking
            inserting_chunks = get_chunks(
                new_videos=new_video_segment,
                chunk_func=self.chunk_func,
                max_token_size=self.chunk_token_size,
            )
            _add_chunk_keys = await self.text_chunks.filter_keys(
                list(inserting_chunks.keys())
            )
            inserting_chunks = {
                k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
            }
            if not len(inserting_chunks):
                logger.warning(f"All chunks are already in the storage")
                await self._insert_done()
                return
            logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")
            if self.enable_naive_rag:
                logger.info("Insert chunks for naive RAG")
                await self.chunks_vdb.upsert(inserting_chunks)

            # TODO: no incremental update for communities now, so just drop all
            # await self.community_reports.drop()

            # ---------- extract/summary entity and upsert to graph
            logger.info("[Entity Extraction]...")
            extraction_result = await self.entity_extraction_func(
                inserting_chunks,
                knowledge_graph_inst=self.chunk_entity_relation_graph,
                entity_vdb=self.entities_vdb,
                global_config=asdict(self),
            )
            if extraction_result is None:
                logger.warning("No new entities found")
            else:
                maybe_new_kg, _, _ = extraction_result
                self.chunk_entity_relation_graph = maybe_new_kg
            # ---------- commit upsertings and indexing
            await self.text_chunks.upsert(inserting_chunks)
        except Exception:
            logger.exception("VideoRAG insertion failed")
            raise
        else:
            await self._insert_done()

    async def _insert_start(self):
        tasks = []
        for storage_inst in [
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_start_callback())
        await asyncio.gather(*tasks)

    async def _save_video_segments(self):
        tasks = []
        for storage_inst in [
            self.video_segment_feature_vdb,
            self.video_segments,
            self.video_path_db,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)
    
    async def _insert_done(self):
        tasks = []
        for storage_inst in [
            self.text_chunks,
            self.llm_response_cache,
            self.entities_vdb,
            self.chunks_vdb,
            self.chunk_entity_relation_graph,
            self.video_segment_feature_vdb,
            self.video_segments,
            self.video_path_db,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        # These files are committed separately, not in a multi-file transaction.
        # If any save fails, reloading may mix old and new indexes. Do not allow
        # automatic recovery in that case. Wait for every save before returning.
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except BaseException as exc:
            raise UnsafeVideoRAGState("VideoRAG index commit was interrupted; persisted state must be checked") from exc
        errors = [result for result in results if isinstance(result, BaseException)]
        if errors:
            raise UnsafeVideoRAGState("VideoRAG index commit failed; persisted state must be checked") from errors[0]

    async def _query_done(self):
        tasks = []
        for storage_inst in [self.llm_response_cache]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)
