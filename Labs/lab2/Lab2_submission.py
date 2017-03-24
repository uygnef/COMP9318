## import modules here 
import pandas as pd
import numpy as np


################# Question 1 #################

# helper functions
def project_data(df, d):
    # Return only the d-th column of INPUT
    return df.iloc[:, d]

def select_data(df, d, val):
    # SELECT * FROM INPUT WHERE input.d = val
    col_name = df.columns[d]
    return df[df[col_name] == val]

def remove_first_dim(df):
    # Remove the first dim of the input
    return df.iloc[:, 1:]

def slice_data_dim0(df, v):
    # syntactic sugar to get R_{ALL} in a less verbose way
    df_temp = select_data(df, 0, v)
    return remove_first_dim(df_temp)


def buc_rec_optimized(df):  # do not change the heading of the function

    all_list = []

    header = list(df)
    df_out = pd.DataFrame(columns=header)
    output = _buc_rec_optimized(df, [], df_out)
    return df_out


def _buc_rec_optimized(df, pre_num, df_out):  # help function
    # Note that input is a DataFrame
    dims = df.shape[1]

    if dims == 1:
        # only the measure dim
        input_sum = sum(project_data(df, 0))
        pre_num.append(input_sum)

        df_out.loc[len(df_out)] = pre_num

    else:
        # the general case

        dim0_vals = set(project_data(df, 0).values)
        temp_pre_num = deepcopy(pre_num)
        for dim0_v in dim0_vals:
            pre_num = deepcopy(temp_pre_num)
            sub_data = slice_data_dim0(df, dim0_v)
            pre_num.append(dim0_v)

            _buc_rec_optimized(sub_data, pre_num, df_out)
        ## for R_{ALL}
        sub_data = remove_first_dim(df)

        pre_num = deepcopy(temp_pre_num)
        pre_num.append("ALL")
        _buc_rec_optimized(sub_data, pre_num, df_out)



################# Question 2 #################
def v_opt_dp(x, num_bins):  # do not change the heading of the function
    global _x, _num_bins, dp_matrix

    dp_matrix = [[-1 for i in range(len(x))] for j in range(num_bins)]

    _x = x
    _num_bins = num_bins
    _v_opt_dp(0, num_bins - 1)  # bin is 0-3
    for i in range(len(dp_matrix)):  # transfer 0 from float to int
        for j in range(len(dp_matrix[i])):
            if (dp_matrix[i][j] == 0.0):
                dp_matrix[i][j] = int(0)
    return dp_matrix


def _v_opt_dp(mtx_x, remain_bins):  # mtx_x is the index of x, we will put
    # all element behind it to the reamin bin

    global _x, _num_bins, dp_matrix

    if (_num_bins - remain_bins - mtx_x < 2) and (len(_x) - mtx_x > remain_bins):
        _v_opt_dp(mtx_x + 1, remain_bins)
        if (remain_bins == 0):
            dp_matrix[remain_bins][mtx_x] = np.var(_x[mtx_x:]) * len(_x[mtx_x:])
            return

        _v_opt_dp(mtx_x, remain_bins - 1)

        min_list = [dp_matrix[remain_bins - 1][mtx_x + 1]]

        for i in range(mtx_x + 1, len(_x)):
            min_list.append(dp_matrix[remain_bins - 1][i] + (i - mtx_x) * np.var(_x[mtx_x:i]))

        dp_matrix[remain_bins][mtx_x] = min(min_list)