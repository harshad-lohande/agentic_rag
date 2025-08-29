---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	classify_query(classify_query)
	transform_query(transform_query)
	generate_hyde_document(generate_hyde_document)
	web_search(web_search)
	retrieve_docs(retrieve_docs)
	rerank_documents(rerank_documents)
	summarize_history(summarize_history)
	generate_answer(generate_answer)
	safety_check(safety_check)
	web_search_safety_check(web_search_safety_check)
	enter_retrieval_correction(enter_retrieval_correction)
	handle_retrieval_failure(handle_retrieval_failure)
	enter_grounding_correction(enter_grounding_correction)
	smart_retrieval(smart_retrieval)
	hybrid_context(hybrid_context)
	handle_grounding_failure(handle_grounding_failure)
	__end__([<p>__end__</p>]):::last
	__start__ --> classify_query;
	classify_query -. &nbsp;retrieve&nbsp; .-> retrieve_docs;
	classify_query -.-> transform_query;
	enter_grounding_correction -.-> handle_grounding_failure;
	enter_grounding_correction -.-> hybrid_context;
	enter_grounding_correction -.-> smart_retrieval;
	enter_retrieval_correction -.-> generate_hyde_document;
	enter_retrieval_correction -.-> handle_retrieval_failure;
	enter_retrieval_correction -.-> transform_query;
	enter_retrieval_correction -.-> web_search;
	generate_answer -.-> safety_check;
	generate_answer -.-> web_search_safety_check;
	generate_hyde_document --> retrieve_docs;
	hybrid_context --> generate_answer;
	rerank_documents -.-> enter_retrieval_correction;
	rerank_documents -. &nbsp;summarize&nbsp; .-> summarize_history;
	retrieve_docs -.-> enter_retrieval_correction;
	retrieve_docs -.-> rerank_documents;
	safety_check -. &nbsp;END&nbsp; .-> __end__;
	safety_check -.-> enter_grounding_correction;
	smart_retrieval --> generate_answer;
	summarize_history --> generate_answer;
	transform_query --> retrieve_docs;
	web_search --> summarize_history;
	handle_grounding_failure --> __end__;
	handle_retrieval_failure --> __end__;
	web_search_safety_check --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
