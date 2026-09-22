

#하락한 목록들의 리스트 찾기

def find_regression_list(config_name:str,gold_dict:dict,accepted_dict:dict,acceptable:float) -> list:

    failed_list=[]
    for ele in gold_dict:
        value=gold_dict[ele] - accepted_dict[ele]
        if value>acceptable:
            failed_list.append({"config":config_name,"metric":ele,"baseline":gold_dict[ele],"current":accepted_dict[ele],"drop":value})

    return failed_list

