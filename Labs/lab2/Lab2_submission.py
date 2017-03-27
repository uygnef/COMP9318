## import modules here 
import pandas as pd
import numpy as np
import helper

################# Question 1 #################

# you can call helper functions through the helper module (e.g., helper.slice_data_dim0)

def buc_rec_optimized(df):  # do not change the heading of the function

    if df.shape[0] == 1:
        df_out = one_dim_duc(df)
    else:
        header = list(df)
        df_out = pd.DataFrame(columns=header)
        _buc_rec_optimized(df, [], df_out)
    return df_out


def deepcopy(list):
    b = []
    for i in list:
        b.append(i)
    return b


def read_data(filename):
    df = pd.read_csv(filename, sep='\t')
    return (df)


def output(val):
    print('=>\t{}'.format(val))


def one_dim_duc(df):
    vals = list(df.loc[0])
    result = [vals[:-1]]
    for i, val in enumerate(result):
        temp = deepcopy(val)
        for j, list_val in enumerate(temp):
            if list_val != 'ALL':
                temp2 = deepcopy(temp)
                temp2[j] = 'ALL'
                if set(temp2) != {'ALL'}:
                    result.append(temp2)
    result.append(['ALL' for _ in vals[:-1]])
    fin_result=[]
    for i in result:
        i.append(vals[-1])
        if i not in fin_result:
            fin_result.append(i)

    result = pd.DataFrame(fin_result, columns=list(df))
    return result


def _buc_rec_optimized(df, pre_num, df_out):  # help function
    # Note that input is a DataFrame
    dims = df.shape[1]

    if dims == 1:
        # only the measure dim
        input_sum = sum(helper.project_data(df, 0))
        pre_num.append(input_sum)

        df_out.loc[len(df_out)] = pre_num

    else:
        # the general case

        dim0_vals = set(helper.project_data(df, 0).values)
        temp_pre_num = deepcopy(pre_num)
        for dim0_v in dim0_vals:
            pre_num = deepcopy(temp_pre_num)
            sub_data = helper.slice_data_dim0(df, dim0_v)
            pre_num.append(dim0_v)

            _buc_rec_optimized(sub_data, pre_num, df_out)
        ## for R_{ALL}
        sub_data = helper.remove_first_dim(df)

        pre_num = deepcopy(temp_pre_num)
        pre_num.append("ALL")
        _buc_rec_optimized(sub_data, pre_num, df_out)


    ################# Question 2 #################

def v_opt_dp(x, num_bins):# do not change the heading of the function

    global _x, _num_bins, dp_matrix, dp_index

    dp_matrix = [[-1 for i in range(len(x))] for j in range(num_bins)]
    dp_index = [[-1 for i in range(len(x))] for j in range(num_bins)]
    _x = x
    _num_bins = num_bins
    _v_opt_dp(0, num_bins-1) #bin is 0-3

    start = dp_index[-1][0]
    pre_start = start
    bins = [x[:start]]
    for i in range(len(dp_index)-2, 0, -1):
        start = dp_index[i][start]
        bins.append(x[pre_start:start])
        pre_start = start
    bins.append(x[pre_start:])
    return dp_matrix, bins

def _v_opt_dp(mtx_x, remain_bins): #mtx_x is the index of x, we will put
                                    #all element behind it to the reamin bin
    
    global _x, _num_bins, dp_matrix, dp_index
    
    if( _num_bins - remain_bins - mtx_x < 2) and (len(_x) - mtx_x > remain_bins):
        _v_opt_dp(mtx_x+1, remain_bins)
        if(remain_bins == 0):
            dp_matrix[remain_bins][mtx_x] = np.var(_x[mtx_x:])*len(_x[mtx_x:])
            return 

        _v_opt_dp(mtx_x, remain_bins - 1)  

        min_list = [dp_matrix[remain_bins-1][mtx_x+1]]

        for i in range(mtx_x+2, len(_x)):
            min_list.append(dp_matrix[remain_bins-1][i] + (i - mtx_x)*np.var(_x[mtx_x:i])) 

        dp_matrix[remain_bins][mtx_x] = min(min_list)
        dp_index[remain_bins][mtx_x] = min_list.index(min(min_list)) + mtx_x +1

input_data = read_data('./c_.txt')
output = buc_rec_optimized(input_data)
print(output)