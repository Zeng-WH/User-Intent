import rouge

# from run_seq2seq import main
rouge = rouge.Rouge()

def compute_rouges(sources, targets):
    """计算rouge-1、rouge-2、rouge-l

    Args:
        sources (List[str]): prediction, 注意不包含空格
        targets (List[str]): groundtruth, 注意不包含空格

    Returns:
        Dict[str: float]: rouge scores, {
            "rouge-1": 0.0,
            "rouge-2": 0.0,
            "rouge-l": 0.0
        }
    """

    
    list_of_hypotheses = []
    list_of_references = []
    for i in range(len(sources)):
        if sources[i] and targets[i]:
            #print(sources[i].strip())
            list_of_hypotheses.append(sources[i].strip()) # 计算rouge时要手动在字间增加空格
            list_of_references.append(targets[i].strip())
        else:
            print("Warning: there is empty string when computing rouge scores")
    scores = rouge.get_scores(list_of_hypotheses, list_of_references, avg=True)
    return {
        "rouge-1-p": scores["rouge-1"]["p"],
        "rouge-2-p": scores["rouge-2"]["p"],
        "rouge-l-p": scores["rouge-l"]["p"],
        "rouge-1-r": scores["rouge-1"]["r"],
        "rouge-2-r": scores["rouge-2"]["r"],
        "rouge-l-r": scores["rouge-l"]["r"],
        "rouge-1-f": scores["rouge-1"]["f"],
        "rouge-2-f": scores["rouge-2"]["f"],
        "rouge-l-f": scores["rouge-l"]["f"],

    }
    



    '''
    total_R1_P = 0
    total_R1_R = 0
    total_R1_F = 0
    total_R2_P = 0
    total_R2_R = 0
    total_R2_F = 0
    total_RL_P = 0
    total_RL_R = 0
    total_RL_F = 0
    for i in range(len(sources)):
        rouge_score = rouge.get_scores(sources[i], targets[i])
        R_1 = rouge_score[0]["rouge-1"]
        R_2 = rouge_score[0]["rouge-2"]
        R_L = rouge_score[0]["rouge-l"]
        P_R_1 = R_1['p']
        R_R_1 = R_1['r']
        F_R_1 = R_1['f']
        P_R_2 = R_2['p']
        R_R_2 = R_2['r']
        F_R_2 = R_2['f']
        P_R_L = R_L['p']
        R_R_L = R_L['r']
        F_R_L = R_L['f']
        total_R1_P += P_R_1
        total_R1_R += R_R_1
        total_R1_F += F_R_1
        total_R2_P += P_R_2
        total_R2_R += R_R_2
        total_R2_F += F_R_2
        total_RL_P += P_R_L
        total_RL_R += R_R_L
        total_RL_F += F_R_L
    return{
        "rouge-1-p": total_R1_P/len(sources),
        "rouge-1-r": total_R1_R/len(sources),
        "rouge-1-f": total_R1_F/len(sources),
        "rouge-2-p": total_R2_P/len(sources),
        "rouge-2-r": total_R2_R/len(sources),
        "rouge-2-f": total_R2_F/len(sources),
        "rouge-l-p": total_RL_P/len(sources),
        "rouge-l-r": total_RL_R/len(sources),
        "rouge-l-f": total_RL_F/len(sources),
    }

    '''


if __name__ == "__main__":
    print(compute_rouges(["你好"], ["你好"]))