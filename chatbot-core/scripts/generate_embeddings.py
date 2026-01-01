"""
Script to generate FAISS embeddings from processed chunks.
This creates the vector indices needed for the chatbot backend.
"""
import os
import sys
import numpy as np
import faiss

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.embedding.embed_chunks import embed_chunks
from rag.vectorstore.vectorstore_utils import save_metadata
from utils.logger import LoggerFactory

logger = LoggerFactory.instance().get_logger("generate_embeddings")

def create_faiss_index(embeddings):
    """
    Create a FAISS index from embeddings.
    
    Args:
        embeddings: List of numpy arrays
        
    Returns:
        faiss.Index: The created FAISS index
    """
    # Convert to numpy array
    embeddings_array = np.array(embeddings).astype('float32')
    
    # Get dimension
    dimension = embeddings_array.shape[1]
    
    # Create index (using L2 distance)
    index = faiss.IndexFlatL2(dimension)
    
    # Add vectors to index
    index.add(embeddings_array)
    
    return index

def main():
    """Generate and save embeddings for all processed chunks."""
    logger.info("Starting embedding generation...")
    
    # Get embedded chunks
    embeddings, metadata = embed_chunks(logger)
    
    if len(embeddings) == 0 or len(metadata) == 0:
        logger.error("No embeddings generated. Check if processed chunks exist.")
        return 1
    
    logger.info(f"Generated {len(embeddings)} embeddings")
    
    # Save to vector store
    output_dir = os.path.join(
        os.path.dirname(__file__),
        "..", "data", "embeddings"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # Save for each source type
    sources = set(m.get("metadata", {}).get("source", "unknown") for m in metadata)
    logger.info(f"Found sources: {sources}")
    
    for source in sources:
        # Filter embeddings and metadata for this source
        source_indices = [i for i, m in enumerate(metadata) if m.get("metadata", {}).get("source") == source]
        source_embeddings = [embeddings[i] for i in source_indices]
        source_metadata = [metadata[i] for i in source_indices]
        
        if source_embeddings:
            index_path = os.path.join(output_dir, f"{source}_index.idx")
            metadata_path = os.path.join(output_dir, f"{source}_metadata.pkl")
            
            # Create FAISS index from embeddings
            index = create_faiss_index(source_embeddings)
            
            # Save index and metadata
            faiss.write_index(index, index_path)
            logger.info(f"Saved FAISS index to {index_path}")
            
            save_metadata(source_metadata, metadata_path, logger)
            
            logger.info(f"Saved {len(source_embeddings)} embeddings for source '{source}'")
    
    logger.info("Embedding generation complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
