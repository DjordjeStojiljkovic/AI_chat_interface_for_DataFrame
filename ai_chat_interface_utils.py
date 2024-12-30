# Extracting results function
def extracting_results(original_input : str, result_from_llm : str):
  inds = result_from_llm.find('%start')
  result_from_llm = result_from_llm[:inds] + result_from_llm[inds+len('%start'):]
  inds = result_from_llm.find('%end')
  result_from_llm = result_from_llm[:inds] + result_from_llm[inds+len('%end'):]
  ind1 = result_from_llm.find('%start') + len('%start')
  ind2 = result_from_llm.find('%end')
  return result_from_llm[ind1:ind2].strip()
