# 1. Re-cluster with fine-tuned embeddings                                                          
uv run python3 cluster_papers.py --method embeddings_ft                                             
uv run python3 cluster_papers.py --method louvain_embeddings_ft                                     
                                                                                                      
# 2. Re-run BERTopic with fine-tuned embeddings                                                     
uv run python3 topic_model.py --ft                                                                  
                                                                                                      
# 3. Regenerate paper pages
uv run python3 export_paper_pages.py                                                                
                                                                                                      
# 4. Regenerate the graph HTML                                                                      
uv run python3 export_html.py