import json
import logging
import os
from typing import Any, Dict, List, Optional
import numpy as np

logger = logging.getLogger("omnivision.retrieval")

DEFAULT_DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://rag_user:rag_password@localhost:5433/omnivision_db"
)


class VectorDatabaseManager:
    """Manages PostgreSQL vector storage, HNSW indexing and semantic similarity search."""

    def __init__(self, db_url: str = DEFAULT_DATABASE_URL):
        self.db_url = db_url
        self.conn = None
        self.is_memory_fallback = False
        self._memory_store: List[Dict[str, Any]] = []
        self._connect()

    def _connect(self):
        """Attempts connection to PostgreSQL, falling back to in-memory store if offline."""
        try:
            import psycopg2
            from pgvector.psycopg2 import register_vector

            self.conn = psycopg2.connect(self.db_url)
            self.conn.autocommit = True
            register_vector(self.conn)
            self._initialize_schema()
            logger.info("Connected to PostgreSQL with pgvector extension.")
        except Exception as exc:
            logger.warning(
                "PostgreSQL pgvector unavailable (%s). Initializing in-memory fallback store.",
                exc
            )
            self.is_memory_fallback = True

    def _initialize_schema(self):
        """Creates the vector extension, video events table, and HNSW index."""
        if self.is_memory_fallback or self.conn is None:
            return

        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS video_events (
                    id SERIAL PRIMARY KEY,
                    video_name VARCHAR(255) NOT NULL,
                    storage_path VARCHAR(512) NOT NULL,
                    timestamp_sec FLOAT NOT NULL,
                    scene_embedding vector(512) NOT NULL,
                    metadata JSONB DEFAULT '{}'::jsonb,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)
            # Create HNSW index for sub-millisecond approximate nearest neighbor search
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_video_events_hnsw 
                ON video_events USING hnsw (scene_embedding vector_cosine_ops) 
                WITH (m = 16, ef_construction = 64);
            """)

    def insert_event(
        self,
        video_name: str,
        storage_path: str,
        timestamp_sec: float,
        scene_embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Inserts indexed video frame and embedding vector into the database."""
        if metadata is None:
            metadata = {}

        if self.is_memory_fallback or self.conn is None:
            event_id = len(self._memory_store) + 1
            self._memory_store.append({
                "id": event_id,
                "video_name": video_name,
                "storage_path": storage_path,
                "timestamp_sec": timestamp_sec,
                "scene_embedding": np.array(scene_embedding, dtype=np.float32),
                "metadata": metadata,
            })
            return event_id

        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO video_events (video_name, storage_path, timestamp_sec, scene_embedding, metadata)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id;
            """, (video_name, storage_path, timestamp_sec, scene_embedding, json.dumps(metadata)))
            row = cur.fetchone()
            return row[0] if row else 0

    def search_similar_frames(
        self,
        query_embedding: List[float],
        top_k: int = 4,
    ) -> List[Dict[str, Any]]:
        """Queries for closest keyframes using cosine distance operator (<=>).
        
        Returns:
            List of dictionaries sorted by descending cosine similarity.
        """
        if self.is_memory_fallback or self.conn is None:
            return self._search_in_memory(query_embedding, top_k)

        with self.conn.cursor() as cur:
            # (1 - (scene_embedding <=> query_vec)) gives Cosine Similarity in [0.0, 1.0]
            sql = """
                SELECT id, video_name, storage_path, timestamp_sec, metadata,
                       (1 - (scene_embedding <=> %s::vector)) AS similarity
                FROM video_events
                ORDER BY scene_embedding <=> %s::vector ASC
                LIMIT %s;
            """
            cur.execute(sql, (query_embedding, query_embedding, top_k))
            rows = cur.fetchall()

        results = []
        for r in rows:
            results.append({
                "id": r[0],
                "video_name": r[1],
                "storage_path": r[2],
                "timestamp_sec": float(r[3]),
                "metadata": r[4] if isinstance(r[4], dict) else {},
                "similarity": round(float(r[5]), 4),
            })
        return results

    def _search_in_memory(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Fallback in-memory cosine search when PostgreSQL is not running."""
        if not self._memory_store:
            return []

        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        scored = []

        for item in self._memory_store:
            target_vec = item["scene_embedding"]
            t_norm = np.linalg.norm(target_vec)
            if q_norm == 0 or t_norm == 0:
                sim = 0.0
            else:
                sim = float(np.dot(q_vec, target_vec) / (q_norm * t_norm))
            scored.append({
                "id": item["id"],
                "video_name": item["video_name"],
                "storage_path": item["storage_path"],
                "timestamp_sec": item["timestamp_sec"],
                "metadata": item["metadata"],
                "similarity": round(sim, 4),
            })

        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:top_k]

    def count_events(self) -> int:
        """Returns total number of indexed frame events in store."""
        if self.is_memory_fallback or self.conn is None:
            return len(self._memory_store)

        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM video_events;")
            return cur.fetchone()[0]
