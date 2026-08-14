--> lote de documento
--tratamento paralelo sequencial


1) data_Extraction():

possible for the data:
- ocr
- metadata 
- llm extraction 


I) classify_registry()
-- each event has a descripton of what is it and what usually contains in its documents to be passed to the LLM
-- Each Type event has a diferent types of data to be used, create python fonction for each type 

classify_data()
-- If failed try again, at least 3 times
--- no success --> register the cause and designate to human
-- Check each field it is in original text (text extraction or ocr) --> Obs: if was removed by metadata it should be strict, if it is OCR it should flag possible errors based on the probability
-- if something is flagged as critic () like a number mark it and ask for a human review, for now just flag dont put HUMAN in the loop
-- carry on score of ocr and metadata for each value, classification, so everything that is not the schema and was inputted by the llm should carry a type of score 
-- each field should also carry-on page of the file, name_file and method 
repair_data()
--- Date repair(mapping  of used format and use specific script to fix)
--- Currency repair
--- ISIN
--- Everything in the format of gold records and if it is not defined there, create a standard

3) verificar o registro
verify_registry()
gives back the result: 1 error , flag what was wrong: type, missing or wrong

4) create_report()  create the report based on the ouput of the previous stages, assemble them: verify_registry() + 