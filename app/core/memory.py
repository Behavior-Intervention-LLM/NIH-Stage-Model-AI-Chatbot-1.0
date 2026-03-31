import os
import sqlite3
from typing import List, Optional
from app.core.types import SessionState, Message, MessageRole

class MemoryManager:
    """Hybrid Memory Manager: Handles active SessionState & persistent Local Storage."""
    
    def __init__(self, short_term_limit: int = 20, summary_threshold: int = 10, storage_dir: str = "./local_storage"):
        # --- Active Session Limits ---
        self.short_term_limit = short_term_limit
        self.summary_threshold = summary_threshold
        
        # --- Persistent Storage Setup ---
        self.storage_dir = storage_dir
        self.db_path = os.path.join(self.storage_dir, "app_memory.db")
        os.makedirs(self.storage_dir, exist_ok=True)
        self._initialize_db()

    def _initialize_db(self):
        """Creates the secure, isolated tables for users if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()

    # ==========================================
    # 1. PERSISTENT STORAGE (Disk/SQLite)
    # ==========================================

    def save_message_to_db(self, user_id: str, message: Message):
        """Saves a message to the user's private, long-term history."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO chat_logs (user_id, role, content) VALUES (?, ?, ?)",
            (user_id, message.role.value, message.content)
        )
        conn.commit()
        conn.close()

    def save_uploaded_document(self, user_id: str, file_name: str, file_content: bytes) -> str:
        """Saves a user's research draft to an isolated, private local folder."""
        user_folder = os.path.join(self.storage_dir, "users", user_id)
        os.makedirs(user_folder, exist_ok=True)
        
        safe_filename = os.path.basename(file_name)
        file_path = os.path.join(user_folder, safe_filename)
        
        with open(file_path, "wb") as f:
            f.write(file_content)
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO user_documents (user_id, filename, filepath) VALUES (?, ?, ?)",
            (user_id, safe_filename, file_path)
        )
        conn.commit()
        conn.close()
        
        return file_path

    # ==========================================
    # 2. CHAT RETRIEVAL (NEW)
    # ==========================================

    def get_chat_history(self, user_id: str, limit: int = 50, offset: int = 0) -> List[Message]:
        """Retrieve chat history for a user from the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                """
                SELECT role, content, timestamp 
                FROM chat_logs 
                WHERE user_id = ? 
                ORDER BY timestamp ASC
                LIMIT ? OFFSET ?
                """,
                (user_id, limit, offset)
            )
            
            rows = cursor.fetchall()
            conn.close()
            
            messages = [
                Message(role=MessageRole(role), content=content)
                for role, content, timestamp in rows
            ]
            
            return messages
            
        except Exception as e:
            print(f"Error retrieving chat history: {e}")
            return []

    def search_chat_history(self, user_id: str, query: str) -> list:
        """Search user's chat history by content."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                """
                SELECT role, content, timestamp
                FROM chat_logs
                WHERE user_id = ? AND content LIKE ?
                ORDER BY timestamp DESC
                """,
                (user_id, f"%{query}%")
            )
            
            rows = cursor.fetchall()
            conn.close()
            
            return rows
            
        except Exception as e:
            print(f"Error searching chat history: {e}")
            return []

    # ==========================================
    # 3. FILE MANAGEMENT (NEW)
    # ==========================================

    def get_user_documents(self, user_id: str, limit: int = 50) -> List[dict]:
        """Get all documents uploaded by a user."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                """
                SELECT id, filename, filepath, timestamp
                FROM user_documents
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (user_id, limit)
            )
            
            rows = cursor.fetchall()
            conn.close()
            
            documents = [
                {
                    "id": row[0],
                    "filename": row[1],
                    "filepath": row[2],
                    "timestamp": row[3]
                }
                for row in rows
            ]
            
            return documents
            
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []

    def get_document_by_id(self, user_id: str, doc_id: int) -> Optional[dict]:
        """Retrieve a specific document by ID (with user isolation for security)."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                """
                SELECT id, filename, filepath, timestamp
                FROM user_documents
                WHERE user_id = ? AND id = ?
                """,
                (user_id, doc_id)
            )
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    "id": row[0],
                    "filename": row[1],
                    "filepath": row[2],
                    "timestamp": row[3]
                }
            else:
                return None
            
        except Exception as e:
            print(f"Error retrieving document: {e}")
            return None

    # ==========================================
    # 4. ACTIVE SESSION LOGIC (RAM)
    # ==========================================

    def get_short_term_memory(self, state: SessionState) -> List[Message]:
        """Gets recent messages from the active session."""
        return state.get_recent_messages(self.short_term_limit)
    
    def get_summary(self, state: SessionState) -> Optional[str]:
        return state.summary
    
    def should_summarize(self, state: SessionState) -> bool:
        return len(state.messages) >= self.summary_threshold and state.summary is None
    
    def create_summary(self, state: SessionState) -> str:
        """Placeholder for LLM summarization."""
        recent_messages = self.get_short_term_memory(state)
        summary_parts = []
        
        for msg in recent_messages[:10]:
            if msg.role == MessageRole.USER:
                summary_parts.append(f"User: {msg.content[:100]}")
            elif msg.role == MessageRole.ASSISTANT:
                summary_parts.append(f"Assistant: {msg.content[:100]}")
        
        return "\n".join(summary_parts)
    
    def update_summary(self, state: SessionState, summary: str):
        state.summary = summary
    
    def get_context_for_agent(self, state: SessionState, include_summary: bool = True) -> str:
        """Formats the active state (slots, summary, recent chat) into a prompt string."""
        context_parts = []
        
        if include_summary and state.summary:
            context_parts.append(f"Summary:\n{state.summary}\n")
        
        slots_info = []
        if state.slots.need_stage is not None:
            slots_info.append(f" Needs Stage: {state.slots.need_stage}")
        if state.slots.stage:
            slots_info.append(f" Detected Stage: {state.slots.stage} (confidence: {state.slots.stage_confidence:.2f})")
        if state.slots.user_goal:
            slots_info.append(f" User Goal: {state.slots.user_goal}")
        
        if slots_info:
            context_parts.append(f"Extracted Context:\n" + "\n".join(slots_info) + "\n")
        
        recent = self.get_short_term_memory(state)
        if recent:
            messages_text = "\n".join([
                f"{msg.role.value}: {msg.content}" 
                for msg in recent[-5:]
            ])
            context_parts.append(f"Recent Conversation:\n{messages_text}")
        
        return "\n".join(context_parts)

# Singleton initialization
memory_manager = MemoryManager()