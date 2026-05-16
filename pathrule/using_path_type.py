#
# License: Apache 2.0
#
from bandit.core.context import Context
import ast
import io
import bandit
#import hello
#import game_analysis
#import vcg_analysis
from bandit.core import issue
from bandit.core import test_properties as test

import networkx as nx
import pylab
import matplotlib

import pandas as pd 
import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy.optimize import linprog

import pandas as pd
import numpy as np
import itertools
import nashpy

import json
from _ast import If

from networkx.drawing.nx_agraph import graphviz_layout
import time

r"""
The T.B.Cs one sees in comments are for my post-PhD research directions, with the base cases for the PhD deemed to be in place.
T.B.C
1. global variable to store state that can say whether the path variable was used without the escaping
2. creation of graph
3. game design and analysis
4. modify code base with secure code

Post PhD - unclear if a plugin like this can be split into different modules. The Bandit plugin
seems to want all implementation in one function. Wasn't an important task for the research itself,
but for better usability, modularization should be figured out and implemented in future by any one interested
in the implementation aspects
"""
str_counter=0
call_counter=0
current_file=""
my_global_graph = nx.DiGraph()#to show the chain of flows - starting from the first file, checking if it's using the sec lib, and then going to the second file, and so on
my_global_graph_for_inout = nx.DiGraph()#to link the files not using sec lib, basically is another/hub-and-spoke view of the global graph
my_global_graph_secure = nx.DiGraph()#to link the files using the sec lib
my_global_graph_undirected = nx.Graph()#to get the average path, as the graph for input validation type issues
files_processed_so_far = []
#Mainly to help with the experiment.
expected_list_of_files = ['commonvalidator.py', 'db.py',  'hello.py',
                          'hello - Copy.py', 'hello - Copy (2).py', 'hello - Copy (3).py', 'hello - Copy (4).py', 'hello - Copy (5).py', 'hello - Copy (6).py',
                          'hello - Copy (7).py', 'hello - Copy (8).py', 'hello - Copy (9).py', 'hello - Copy (10).py', 'hello - Copy (11).py', 
                          'hello - Copy (12).py', 'hello - Copy (13).py', 'hello - Copy (14).py', 'hello - Copy (15).py', 'hello - Copy (16).py', 
                          'hello - Copy (17).py', 'hello - Copy (18).py', 'hello - Copy (19).py', 'hello - Copy (20).py']
#the smaller list for a smaller sample
r"""
expected_list_of_files = ['commonvalidator.py', 'db.py', 'hello - Copy (16).py', 'hello - Copy (17).py', 'hello - Copy (18).py', 'hello - Copy (19).py', 'hello - Copy (20).py', 'hello - Copy.py', ]
"""
pass_number = 0
finxed_line = "name=up.using_path(firstname)"

# 1. Define a sample convex function for the complex analysis
def convex_function(x):
    return x**2

def solve_zerosum_with_linprog(U):
    '''solve_zerosum_with_linprog(): Solve a zero sum game using linear programming
    
        INPUT: U (k*k square matrix), payoffs in zero sum game (opponent gets -U.T)
        OUTPUT: alpha (k-vector) of probability weights for each action (the symmetric equilibrium)
        
        Source: https://github.com/GamEconCph/2023-lectures/blob/main/Bayesian%20Games/bimatrix.py
    '''
    k, k2 = U.shape
    assert k == k2, f'Input matrix must be square, got {k}*{k2}'

    oo = np.zeros((1,k))
    ii = np.ones((1,k))

    # objective: c = [-1, 0, 0, ..., 0]
    c = np.insert(oo, 0, -1.0) # insert -1 in front (pos = index 0)
    
    # inequality constraints A*x <= b
    # top = [ 1 ...
    #         1 -1*A.Tl
    #         1  ...  ]
    # bot = [ 0 -1 0 0 
    #         0 0 -1 0 
    #         0 0 0 -1]
    top  = np.hstack( (ii.T, -1*U.T) )
    bot  = np.hstack( (oo.T, -1*np.eye(k)) )
    A_ub = np.vstack((top, bot))
    
    b_ub = np.zeros((1, 2*k))
    b_ub = np.matrix(b_ub)
    
    # contraints Ax = b
    # A = [0, 1, 1, ..., 1]
    A = np.matrix(np.hstack((0, np.ones((k,)))))
    b = 1.0 # just one condition so scalar 

    # v and alpha must be non-negative
    bounds = [(0,None) for i in range(k+1)]

    # call the solver
    sol = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A, b_eq=b)
    
    # remove the first element: just return the Nash EQ 
    alpha = sol.x[1:]
    return alpha

def best_response(U, i): 
    """best_response(): 
        INPUTS: 
            U: list of payoff matrices 
            i: (int) player for whom to do the best response 

        OUTPUT: 
            br: (NEQ*2) matrix, where br[:,0] is opponent strategies
                and br[:,1] are the best responses. If one strategy a
                has multiple best responses, then there will be several
                columns in br with br[:,0]==a. 
                
        SOURCE:
        https://github.com/GamEconCph/2023-lectures/blob/main/Bayesian%20Games/bimatrix.py
    """
    j = 1-i # opponent
    if i == 0: 
        Ui = U[0]
    elif i == 1: 
        Ui = U[1].T # so that i becomes row player 
    else: 
        raise Exception(f'Not implemented for n>2 players, got i={i}')

    nai, naj = Ui.shape

    # initialie 
    br = []

    for aj in range(naj):
        # column of U corresponding to the aj'th action of the opponent
        Ui_j = Ui[:, aj] 

        # find index values for the rows where Ui_j attains the max
        idim = 0 # there will not be more dimensions in our case 
        br_ij = np.where(Ui_j == Ui_j.max())[idim]

        for b in br_ij: 
            br.append([aj, b])

    return np.array(br)
    
def print_payoffs(U, A, round_decimals=None): 
    '''print_payoffs: Nicely formatted for a 2*2 game 
        INPUTS: 
            U1,U2: (matrices, dim=na1*na2) Payoffs 
            A1: (list of str, len=na1) List of actions of player 1
            A2: (list of str, len=na2) list of actions of player 2
            round_decimals: (int) Number of decimals of precision to print with 
        
        OUTPUT:
            tab: pandas dataframe, na1*na2 with payoff tuples 
            
        SOURCE:
        https://github.com/GamEconCph/2023-lectures/blob/main/Bayesian%20Games/bimatrix.py
    '''
    assert len(U) == 2, f'only implemented for 2-player games'
    assert len(A) == 2, f'only implemented for 2-player games'

    U1 = U[0]
    U2 = U[1]
    A1 = A[0]
    A2 = A[1]

    print("------printing from within the print_payoffs function--------")
    print(U1)
    print(U2)
    print(A1)
    print(A2)
    

    na1,na2 = U1.shape
    print("------na1------")
    print(na1)
    print("------na1------")
    print("------na2------")
    print(na2)
    print("------na2------")
    assert len(A1) == na1
    assert len(A2) == na2

    if not (round_decimals is None):
        assert np.isscalar(round_decimals), f'round_decimals must be an integer' 
        print(type(round_decimals))
        #U1 = U1.round(round_decimals)
        #U2 = U2.round(round_decimals)

    # "matrix" of tuples 
    #X = [[("unknown","unknown") if (((A1[r]=="U") and ((A2[c]=="L") or (A2[c]=="LL"))) or ((A1[r]=="D") and ((A2[c]=="R") or (A2[c]=="RR")))) else (U1[r,c],U2[r,c]) for c in range(na2)] for r in range(na1)]
    #matrix navigation in action
    X = [[("unknown","unknown") if ((A1[r]=="D") and ((A2[c]=="R") or (A2[c]=="RR"))) else (U1[r,c],U2[r,c]) for c in range(na2)] for r in range(na1)]
    #X = [[(U1[r,c],U2[r,c]) for c in range(na2)] for r in range(na1)]
    print("------X------")
    print(X)
    print("------X------")

    # dataframe version 
    tab = pd.DataFrame(X, columns=A2, index=A1)
    print("------printing from within the print_payoffs function--------")
    return tab 

def find_undominated_actions(U_in, i, A, DOPRINT=False):
    '''find_undominated_actions: finds the actions for player i that are
        not strictly dominated by another action
        
        INPUTS: 
            U_in: (matrix, na1*na2) Payoffs (player 1, player 2)
            i: (integer) Which player we are currently examining
            A: (list) List of actions (len = # of actions for this player)
            
        OUTPUT: 
            AA: (list) undominated actions 
            IA: (list of integers) integers i s.t. AA = [A[i] for i in IA]
            ANYDOMINATED: (bool) True if at least one action was strictly dominated
            
        Source: https://github.com/GamEconCph/2023-lectures/blob/main/Bayesian%20Games/bimatrix.py
    '''
    
    AA = []
    IA = []
    nA = len(A)
    
    # 1. ensure that U has actions of player i along 0th dimension 
    if i == 0: 
        # 1.a already the case 
        U = np.copy(U_in)
    else: 
        # 1.b transpose 
        U = U_in.T 
    print("----U in find undominated actions-----")
    print(U)
    print("----U in find undominated actions-----")
    # 2. determine if each action has other dominated actions 
    for ia in range(nA):
        DOMINATED = False 
        UNKNOWN = False
        UNK1 = []
        UNK2 = []
                
        for ia_ in range(nA): 
            # 2.a loop through all *other* strategies 
            print("ia")
            print(ia)
            print("U[ia]")
            print(U[ia])
            print("ia_")
            print(ia_)
            print("U[ia_]")
            print(U[ia_])
            if ia_ == ia: 
                continue
            indices = np.where(U == "unknown")
            print(type(indices))
            num_indices = len(indices)
            if(indices[num_indices-1]):
                UNKNOWN = True
                print("num_indices",num_indices)
                print("indices", indices)
                print("All indices of unknown:", num_indices)
                print(indices[num_indices-1])
                unknown_at = indices[num_indices-1][0]
                print("Unknown at", unknown_at)
                UNK1=np.delete(U[ia],unknown_at)
                UNK2=np.delete(U[ia_],unknown_at)
            # 2.b check if ia_ always gives a higher payoff than ia (i.e. domination)
            if UNKNOWN == False:
                if np.all(U[ia_] > U[ia]):
                    print("found dominated strategy") 
                    print(U[ia])
                    print(U[ia_])
                    print("found dominated strategy")
                    DOMINATED = True
                    break # exit search: enough that we have found one
            elif UNKNOWN == True:
                if np.all(UNK1 > UNK2):
                    print("found dominated strategy") 
                    print(UNK1)
                    print(UNK2)
                    print("found dominated strategy")
                    DOMINATED = True
                    break # exit search: enough that we have found one
        if UNKNOWN == False:
        # 2.c append or not 
            if not DOMINATED: 
                AA.append(A[ia])
                IA.append(ia)
        elif UNKNOWN == True:
            if not DOMINATED: 
                AA.append(UNK2)
                IA.append(ia)
            
    # 3. convenient boolean 
    ANYDOMINATED = (len(AA) < len(A))
    
    return AA,IA,ANYDOMINATED

def IESDS(A, U, DOPRINT=False, maxit=10000): 
    '''Iterated Elimination of Strictly Dominated Strategies 
        INPUTS: 
            A: (list of lists) n lists (one for each player), 
                    each has len = # of actions to player i
            U: (list, len=n) list of na1*na2 matrices of payoffs
            DOPRINT: (bool) whether to print output to terminal 
            maxit: (int) break algorithm if this count is ever reached
                (note: the algorithm is not approximate so we can compute 
                what maxit is in the worst case)
        OUTPUT: Actions and payoffs for the undominated game
            A_undominated: (n-list of vectors) 
            U_undominated: (n-list of matrices of payoffs)
            
        Source: https://github.com/GamEconCph/2023-lectures/blob/main/Bayesian%20Games/bimatrix.py
    '''
    
    U_undominated = copy.copy(U)
    A_undominated = copy.copy(A)
    
    n = len(U)
    na1,na2 = U[0].shape

    # checks 
    assert n == 2, f'Code only implemented for 2-player games '
    assert len(A) == n
    for i in range(n): 
        assert len(A[i]) == U[i].shape[i]
        assert U[i].shape == (na1,na2), f'Payoff matrix for player {i+1} is {U[i].shape}, but {(na1,na2)} for player 1'

    # initialize flags 
    D = np.ones((n,), dtype='bool')
    
    for it in range(maxit): 

        for i in range(n): 
            # find undominated actions 
            A_undominated[i], IA, D[i] = find_undominated_actions(U_undominated[i], i, A_undominated[i], DOPRINT)

            # if we found at least one, remove it/them from the game 
            if D[i]: 
                # remove from both players' payoff matrices 
                for j in range(n): 
                    if i == 0: 
                        U_undominated[j] = U_undominated[j][IA, :]
                    else: 
                        U_undominated[j] = U_undominated[j][:, IA]


        # break once we have run an iteration without finding any strategies to remove 
        if D.any() == False: 
            break

    return A_undominated, U_undominated

def compute_full_matrix(U1, U2, p, action_names=None):
    """      
    "        Source of this method: in the preview tab of https://github.com/GamEconCph/2023-lectures/blob/main/Bayesian%20Games/BNE.ipynb?short_path=6214124
    "        This method has been tweaked to accommodate the theory of conservation of payoffs, which result in jagged matrices/arrays
    "        Assumes that only player 2's type varies \n",
    "        (this means that player 1 has one action per row in U1, \n",
    "         while 2 has nA2**2 (one choice per type))\n",
    "        Both players have one utility matrix for each realization \n",
    "        of player 2's type. \n",
    "         \n",
    "        INPUTS: \n",
    "            U1: list of 2 payoff matrices for player 1 (row player)\n",
    "            U2: list of 2 payoff matrices for player 2 (column player)\n",
    "            p: (scalar) Probability that player 2 is the first type \n",
    "            action_names: [optional] 2-list of names of actions (nA1 and nA2 long)\n",
    "        OUTPUTS: \n",
    "            t1, t2: wide-form payoff matrices suitable for finding the NE \n",
    "            A1, A2: names of actions \n",
    """
    assert len(U1) == 2
    assert len(U2) == 2
    assert np.isscalar(p)
    nA1, nA2 = U1[0].shape
    print("nA1 is",nA1)
    print("nA2 is",nA2)
    #t1 = np.zeros((nA1, nA2*nA2))
    #t2 = np.zeros((nA1, nA2*nA2))
    t1 = np.array([[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,""]])
    t2 = np.array([[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,""]])
    print("t1 is",t1)
    print("t2 is",t2)
    
    for ia1 in range(nA1):
        print("Processing ia1 =",ia1)
        i_col = 0
        
        for a2_1 in range(nA2):
            print("Outer for loop a2_1 =",a2_1)
            for a2_2 in range(nA2):
                print("Inner for loop a2_2",a2_2)
                print("U1[0][ia1,a2_1] is",U1[0][ia1,a2_1])
                print("U1[1][ia1,a2_2] is",U1[1][ia1,a2_2])
                print("U2[0][ia1,a2_1] is",U2[0][ia1,a2_1])
                print("U2[1][ia1,a2_2] is",U2[1][ia1,a2_2])
                """
                Base logic
                t1[ia1,i_col] = p*U1[0][ia1,a2_1] + (1.-p)*U1[1][ia1,a2_2]
                t2[ia1,i_col] = p*U2[0][ia1,a2_1] + (1.-p)*U2[1][ia1,a2_2]
                """                    
                if (((U1[0][ia1,a2_1]) == "unknown") or ((U1[1][ia1,a2_2]) == "unknown")):
                    if (((U1[0][ia1,a2_1]) == "unknown") and not ((U1[1][ia1,a2_2]) == "unknown")):
                        t1[ia1,i_col] = (1.-p)*float(U1[1][ia1,a2_2])
                    elif (not ((U1[0][ia1,a2_1]) == "unknown") and ((U1[1][ia1,a2_2]) == "unknown")):
                        t1[ia1,i_col] = p*float(U1[0][ia1,a2_1])
                    else:
                        print("Looks like we got all unknowns, so skipping assignments altogether")
                        t1[ia1,i_col] = "unknown"
                        #continue
                else:
                    t1[ia1,i_col] = p*float(U1[0][ia1,a2_1]) + (1.-p)*float(U1[1][ia1,a2_2])
                    
                if (((U2[0][ia1,a2_1]) == "unknown") or ((U2[1][ia1,a2_2]) == "unknown")):
                    if (((U2[0][ia1,a2_1]) == "unknown") and not ((U2[1][ia1,a2_2]) == "unknown")):
                        t2[ia1,i_col] = (1.-p)*float(U2[1][ia1,a2_2])
                    elif (not ((U2[0][ia1,a2_1]) == "unknown") and ((U2[1][ia1,a2_2]) == "unknown")):
                        t2[ia1,i_col] = p*float(U2[0][ia1,a2_1])
                    else:
                        print("Looks like we got all unknowns, so skipping assignments altogether")
                        t2[ia1,i_col] = "unknown"
                        #continue
                else:
                    print("Am inside the else in the innermost for loop")
                    print("U2[0][ia1,a2_1] is",U2[0][ia1,a2_1])
                    print("U2[1][ia1,a2_2] is",U2[1][ia1,a2_2])
                    t2[ia1,i_col] = p*float(U2[0][ia1,a2_1]) + (1.-p)*float(U2[1][ia1,a2_2])
                
                i_col +=1
                
    if action_names is None:
        A1 = [f'{i}' for i in range(nA1)]
        A2 = [f'{a}{b}' for a in range(nA2) for b in range(nA2)]
    else:
        assert len(action_names) == 2
        A1 = action_names[0]
        assert len(A1) == nA1, f'Incorrect # of action names'
        a2 = action_names[1]
        assert len(a2) == nA2, f'Incorrect # of action names'
        
        A2 = [f'{a}{b}' for a in a2 for b in a2]
        
    return t1, t2, A1, A2

def print_my_global_graph():
    print("Printing my_global_graph")
    nx.draw(my_global_graph, with_labels=True, font_weight='bold', font_size=20)
    #pylab.show() #Commented for generation of timing data. Uncomment if you need to visualize this graph.

def print_my_global_graph_for_inout():
    
    #This would be the graph that will help with identifying design issues. Envisioned for situations where a secure library
    #is expected to be used, but isn't used in that expected location.
    
    print("Printing my_global_graph_for_inout")
    nx.draw(my_global_graph_for_inout, with_labels=True, font_weight='bold', font_size=20)
    #pylab.show() #Commented for generation of timing data. Uncomment if you need to visualize this graph.
    
def print_my_global_graph_secure():
    print("Printing my_global_graph_secure")
    nx.draw(my_global_graph_secure, with_labels=True, font_weight='bold', font_size=20)
    #pylab.show() #Commented for generation of timing data. Uncomment if you need to visualize this graph.

r"""
A function to calculate and return payoff values based on VCG concepts. Takes 2 parameters - a basis for the vulnerability category,
and a structure of the actual code to be analyzed. For example, speaking of the selfish routing example, we figure out in the graph who
the node operators are, and what the cost of a route is in the presence and absence of that operator. That difference becomes 
the payoff for that operator - could either be a negative or a positive payoff. For our case, the developer and the security players 
are the operators. 
"""
def get_payoffs_from_vcg(cost_without, cost_with):
    vcg_classical_value = ((cost_without)*(-1))-((cost_with)*(-1))
    payoff = vcg_classical_value*(-1)
    return payoff
    
@test.checks("Call")
@test.test_id("B705")
def is_path_there_call(context):
    start_time = time.perf_counter()
    filenamefullpath = context.filename
    filefullpathtokens = filenamefullpath.split("\\")
    filename=filefullpathtokens[(len(filefullpathtokens)-1)]
    print("================================start writing context object================================")
    print("--------","Files processed so far",":",files_processed_so_far,"--------")
    print("Context node type is",type(context.node))
    global call_counter
    call_counter=call_counter+1
    print("call counter is",call_counter)
    contextrepr = str(context.__repr__())
    print("--------","The context object is of the type",type(context))
    print("--------","The contextrepr object is of the type",type(context.__repr__()))
    print("--------","Context",":",contextrepr,"--------")
    print("--------","Call args",":",context.call_args,"--------")
    print("--------","Call args count",":",context.call_args_count,"--------")
    print("--------","Call function name",":",context.call_function_name,"--------")
    print("--------","Call function qual name",":",context.call_function_name_qual,"--------")
    print("--------","Call keywords",":",context.call_keywords,"--------")
    print("--------","Node",":",context.node,"--------")
    print("--------","Filename",":",context.filename,"--------")
    #print("--------","Line Number",":",context.get_lineno_for_call_arg(),"--------")
    print("--------","String val",":",context.string_val,"--------")
    print("--------","Bytes val",":",context.bytes_val,"--------")
    print("--------","String val as escaped bytes",":",context.string_val_as_escaped_bytes,"--------")
    print("--------","Statement",":",context.statement,"--------")
    print("--------","Function def defaults qual",":",context.function_def_defaults_qual,"--------")
    print("--------","The file data is",context.file_data)
    print("--------","The context file data is of the type",type(context.file_data))
    print("================================end writing context object================================")
    
    r"""
    The Context repr is a tricky representation. 
    It cannot be directly loaded via json.loads, as it's not a standard json.
    The below sections clean the json, with the main goal being to get the line number where the issue is found. 
    """
    
    first_curly_bracket_open = contextrepr.find("{")
    last_curly_bracket_close = contextrepr.rfind("}")
    print("First curly bracket open found at index location",first_curly_bracket_open,"and the last curly bracket close found at index",last_curly_bracket_close)
    context_json_str = contextrepr[first_curly_bracket_open:(last_curly_bracket_close+1)]
    context_json_str = context_json_str.replace("'","\"")
    bad_import_curly_bracket_open_approx = 0
    bad_import_curly_bracket_close_approx = 0
    if "imports" in context_json_str:
        bad_import_curly_bracket_open_approx = context_json_str.index("imports")
    if "import_aliases" in context_json_str:
        bad_import_curly_bracket_close_approx = context_json_str.index("import_aliases")
    context_json_str_1=""
    context_json_str_2=""
    context_json_str_3=""
    
    print("Found bad_import_curly_bracket_open_approx at",bad_import_curly_bracket_open_approx)
    print("Found bad_import_curly_bracket_close_approx at",bad_import_curly_bracket_close_approx)
    
    context_json_str_1=context_json_str[:bad_import_curly_bracket_open_approx]
    print("context_json_str_1",context_json_str_1)
    context_json_str_2=context_json_str[bad_import_curly_bracket_open_approx:bad_import_curly_bracket_close_approx]
    print("context_json_str_2 before replace",context_json_str_2)
    context_json_str_2 = context_json_str_2.replace("{","[")
    context_json_str_2 = context_json_str_2.replace("}","]")
    print("context_json_str_2 after replace",context_json_str_2)
    context_json_str_3=context_json_str[bad_import_curly_bracket_close_approx:]
    print("context_json_str_3",context_json_str_3)
    
    context_json_str_prefinal=context_json_str_1+context_json_str_2+context_json_str_3
    context_json_str_prefinal = context_json_str_prefinal.replace("<","\"")
    context_json_str_prefinal = context_json_str_prefinal.replace(">","\"")
    print("--------","context_json_str_prefinal",":",context_json_str_prefinal,"--------")
    
    bad_file_data_dq_open_approx = 0
    bad_file_data_dq_close_approx = 0
    if "file_data" in context_json_str:
        bad_file_data_dq_start_approx = (context_json_str.index("file_data")+len("file_data")+6) #adding 6 at the end to account for standard json chars and skip 5 of them - 2 double quotes around the key, which is followed by a :, the space, and the single quote at the start of the value. We then replace 2 double quotes from that index, as that is the non-standard char for json at this stage.
    context_json_str_4=""
    context_json_str_5=""
    
    context_json_str_4=context_json_str_prefinal[:bad_file_data_dq_start_approx]
    print("context_json_str_4",context_json_str_4)
    context_json_str_5=context_json_str_prefinal[bad_file_data_dq_start_approx:]
    print("context_json_str_5 before replace",context_json_str_5)
    context_json_str_5=context_json_str_5.replace("\"", "'", 2)
    print("context_json_str_5 after replace",context_json_str_5)
    
    context_json_str_final=context_json_str_4+context_json_str_5
    
    context_json = json.loads(context_json_str_final)
    print("--------","context_json",":",context_json,"--------")
    
    target_line_number = (context_json["lineno"]-1)
    
    inloop_file_to_be_loaded = open(context.filename)
    print("Will test opening the file",inloop_file_to_be_loaded)
    for i, line in enumerate(inloop_file_to_be_loaded):
        print("Inside the for loop for line number",i)
        if i == target_line_number:
            print("The issue has been found at line number",target_line_number)
    inloop_file_to_be_loaded.close()
    
    my_graph = nx.DiGraph() 
     
    # Add edges to to the graph object
    # Each tuple represents an edge between two nodes
    my_graph.add_weighted_edges_from([
                            (1,2,3.0), 
                            (1,3,2.0), 
                            (2,4,2.0), 
                            (3,5,3.0), 
                            (2,5,1.0),
                            (3,6,5.0),
                            (4,6,2.0),
                            (5,6,1.0)])
     
    # Draw the resulting graph
    #nx.draw(my_graph, with_labels=True, font_weight='bold')
    #pylab.show()
    shortest_path = nx.dijkstra_path(my_graph, 1, 6)
    shortest_path_distance = nx.shortest_path_length(my_graph, 1, 6, 'weight', 'dijkstra')
    print("The shortest path between 1 and 6, with EF (56), is",shortest_path)
    print("The distance travelled in the shortest path, with EF (56), is",shortest_path_distance)
    
    my_graph_2 = nx.DiGraph() 
     
    # Add edges to to the graph object
    # Each tuple represents an edge between two nodes
    my_graph_2.add_weighted_edges_from([
                            (1,2,3.0), 
                            (1,3,2.0), 
                            (2,4,2.0), 
                            (3,5,3.0), 
                            (2,5,1.0),
                            (3,6,5.0),
                            (4,6,2.0)])
     
    # Draw the resulting graph
    #nx.draw(my_graph_2, with_labels=True, font_weight='bold')
    #pylab.show()

    #bandit.HIGH randomly returns a false positive ("Undefined variable from import") in Eclipse. It needs to be ignored as mentioned in this link, for the reasons also mentioned - https://stackoverflow.com/a/7954059
    if context.call_args:
        for call_arg in context.call_args:
            if call_arg is not None:
                if 'path:' in call_arg:
                    return bandit.Issue(
                        severity=bandit.HIGH,
                        confidence=bandit.HIGH,
                        text="Use of the flask type, path, detected, that can cause XSS. Never accept an entire path from the user. Instead change the data type here to string."
                    )
    
    #checking for an experimental use case
    if (context.call_function_name_qual=='html.escape' and not 'commonvalidator.py' in context.filename):
        print("We want the security logic around html.escape to be in commonvalidator.py, but it was found in", context.filename)
        print("If the outcome of the Game Theoretic analysis is to finx, this plugin will check-in a copy of the code with the logic in commonvalidator.py. It can be a candidate for merging with the main branch.")
        r"""
        1. Call VCG payment calculator with actual values. Values below are dummy ones. (the num of hops can be = the distance from the source to the end validation location, farther it is, more is the num of hops, making it worse for the dev)
        2. With actual payoffs, call game analysis. Values below are dummy ones.
        3. Based on game analysis results, given the equilibriums, finx or don't finx. I.e., modify the code or not. (how to modify the file, from the AST? Look-up.)
        4. For checking the measure of co-operation, any need to calculate Shapley value?
        """

        # Keeping filename unique for starters. Q to ponder: allow duplicate nodes? For capturing any repetition of code blocks?
       
        #Directed and undirected graph long
        #The way to read the global graph - how many times is commonvalidator's logic to be checked in the target file? That number is appended to commonvalidator.py. We chain each usage, and build a graph to showcase the replication
        usepath_this_iter = ''
        global_graph_num_of_edges=my_global_graph.size()
        print('Size of the vulnerable global graph, i.e. the number of edges, is',global_graph_num_of_edges)
        if global_graph_num_of_edges == 0:
            usepath_this_iter = 'commonvalidator.py (0)'
            my_global_graph.add_node(filename, type='notcommonvalidator')
            print("adding the node for", filename)
            my_global_graph.add_node(usepath_this_iter)
            print("adding the node for", usepath_this_iter)
            my_global_graph.add_weighted_edges_from([(filename,usepath_this_iter,2.0)])
            my_global_graph_undirected.add_node(filename, type='notcommonvalidator')
            print("adding the node for", filename)
            my_global_graph_undirected.add_node(usepath_this_iter)
            print("adding the node for", usepath_this_iter)
            my_global_graph_undirected.add_weighted_edges_from([(filename,usepath_this_iter,2.0)])
        elif global_graph_num_of_edges == 1:
            usepath_last_iter = 'commonvalidator.py ('+str(global_graph_num_of_edges-1)+')'
            usepath_this_iter = 'commonvalidator.py ('+str(global_graph_num_of_edges)+')'
            my_global_graph.add_node(filename, type='notcommonvalidator')
            print("adding the node for", filename)
            my_global_graph.add_node(usepath_this_iter)
            print("adding the node for", usepath_this_iter)
            my_global_graph.add_weighted_edges_from([(usepath_last_iter,filename,0.0)])
            my_global_graph.add_weighted_edges_from([(filename,usepath_this_iter,2.0)])
            my_global_graph_undirected.add_node(filename, type='notcommonvalidator')
            print("adding the node for", filename)
            my_global_graph_undirected.add_node(usepath_this_iter)
            print("adding the node for", usepath_this_iter)
            my_global_graph_undirected.add_weighted_edges_from([(usepath_last_iter,filename,0.0)])
            my_global_graph_undirected.add_weighted_edges_from([(filename,usepath_this_iter,2.0)])
        else:
            round_counter=(global_graph_num_of_edges+1)/2
            usepath_last_iter_dir = 'commonvalidator.py ('+str(round_counter-1)+')'
            usepath_this_iter_dir = 'commonvalidator.py ('+str(round_counter)+')'
            usepath_last_iter_undir = 'commonvalidator.py ('+str(global_graph_num_of_edges-2)+')'
            usepath_this_iter_undir = 'commonvalidator.py ('+str(global_graph_num_of_edges)+')'
            my_global_graph.add_node(filename, type='notcommonvalidator')
            print("adding the node for", filename)
            my_global_graph.add_node(usepath_this_iter_dir)
            print("adding the node for", usepath_this_iter_dir)
            my_global_graph.add_weighted_edges_from([(usepath_last_iter_dir,filename,0.0)])
            my_global_graph.add_weighted_edges_from([(filename,usepath_this_iter_dir,2.0)])
            my_global_graph_undirected.add_node(filename, type='notcommonvalidator')
            print("adding the node for", filename)
            my_global_graph_undirected.add_node(usepath_this_iter_undir)
            print("adding the node for", usepath_this_iter_undir)
            my_global_graph_undirected.add_weighted_edges_from([(usepath_last_iter_undir,filename,0.0)])
            my_global_graph_undirected.add_weighted_edges_from([(filename,usepath_this_iter_undir,2.0)])
        
        #Directed graph hub-and-spoke
        my_global_graph_for_inout.add_weighted_edges_from([
                                ('commonvalidator.py',filename,2.0)])
        
        return bandit.Issue(
            severity=bandit.HIGH,
            confidence=bandit.HIGH,
            text="The XSS check is not in the main validator, and is insecure, as it increases the fix and maintenance time for such issues. This pattern will prevent the code from scaling."
        )
    elif (context.call_function_name_qual=='callers.commonvalidator.using_path'):
        usepath_this_iter_2 = ''
        secure_global_graph_num_of_edges=my_global_graph_secure.size()
        print('Size of the vulnerable global graph, i.e. the number of edges, is',secure_global_graph_num_of_edges)
        if secure_global_graph_num_of_edges == 0:
            usepath_this_iter_2 = 'commonvalidator.py (0)'
            my_global_graph_secure.add_node(filename, type='commonvalidator')
            print("adding the node for", filename)
            my_global_graph_secure.add_node(usepath_this_iter_2)
            print("adding the node for", usepath_this_iter_2)
            my_global_graph_secure.add_weighted_edges_from([(filename,usepath_this_iter_2,2.0)])
        elif secure_global_graph_num_of_edges == 1:
            usepath_last_iter_2 = 'commonvalidator.py ('+str(secure_global_graph_num_of_edges-1)+')'
            usepath_this_iter_2 = 'commonvalidator.py ('+str(secure_global_graph_num_of_edges)+')'
            my_global_graph_secure.add_node(filename, type='commonvalidator')
            print("adding the node for", filename)
            my_global_graph_secure.add_node(usepath_this_iter_2)
            print("adding the node for", usepath_this_iter_2)
            my_global_graph_secure.add_weighted_edges_from([(usepath_last_iter_2,filename,0.0)])
            my_global_graph_secure.add_weighted_edges_from([(filename,usepath_this_iter_2,2.0)])
        else:
            round_counter=(global_graph_num_of_edges+1)/2
            usepath_last_iter_dir_2 = 'commonvalidator.py ('+str(round_counter-1)+')'
            usepath_this_iter_dir_2 = 'commonvalidator.py ('+str(round_counter)+')'
            usepath_last_iter_undir_2 = 'commonvalidator.py ('+str(global_graph_num_of_edges-2)+')'
            usepath_this_iter_undir_2 = 'commonvalidator.py ('+str(global_graph_num_of_edges)+')'
            my_global_graph.add_node(filename, type='commonvalidator')
            print("adding the node for", filename)
            my_global_graph.add_node(usepath_this_iter_dir_2)
            print("adding the node for", usepath_this_iter_dir_2)
            my_global_graph.add_weighted_edges_from([(usepath_last_iter_dir_2,filename,0.0)])
            my_global_graph.add_weighted_edges_from([(filename,usepath_this_iter_dir_2,2.0)])
            my_global_graph_undirected.add_node(filename, type='commonvalidator')
            print("adding the node for", filename)
            my_global_graph_undirected.add_node(usepath_this_iter_undir_2)
            print("adding the node for", usepath_this_iter_undir_2)
            my_global_graph_undirected.add_weighted_edges_from([(usepath_last_iter_undir_2,filename,0.0)])
            my_global_graph_undirected.add_weighted_edges_from([(filename,usepath_this_iter_undir_2,2.0)])
    #if context.filename not in files_processed_so_far:
        #print_my_global_graph()
    #The below if condition should be after the above one. Else, if you add the file to the list before, the global graph will never be printed
    if context.filename not in files_processed_so_far:
        files_processed_so_far.append(context.filename)
    num_files_processed = len(files_processed_so_far)
    expected_num_of_files = len(expected_list_of_files)
    print("So far, we have processed",num_files_processed)
    print("We expect to process", expected_num_of_files)
    r"""
    Checking for app.run isn't the most elegant check. The reason for this check is to catch the moment when the plugin has completed
    processing all files. The reason for such a check is to send the final graph to the game logic, and take a decision to finx or not. 
    If done in the plugin, the game will be played again and again for each context and unfinished
    graph, which we don't want. We want it all to happen, after the plugin has done its work. Given our experimental code written in flask,
    the final context we know is going to be a call to app.run. Due to the Ph.D. time constraints, and the 
    fact that the focus of my topic isn't SAST/compilers in general, or bandit in general, I may consider enhancing things post my degree. 
    For now, this condition is sufficient, for intercepting the moment when the plugin has done its bit in processing the files. With this
    "hack", we are now able to ensure that the game is played once after all contexts of all files have been processed.
    """
    if ((num_files_processed == expected_num_of_files) and (context.call_function_name_qual == 'app.run')):
        print("we have processed all the expected files. The files processed are:")
        print(files_processed_so_far)
        print_my_global_graph()
        print_my_global_graph_for_inout()
        print_my_global_graph_secure()
        r"""
        Below values are dummy ones. Next is to send the above graph below, and do the game analysis on our actual graph
        The question to answer - given a graph of some size, which points to how complex the math behind the defensive code has become,
        can our game analysis work around it, and suggest simpler designs. For our experiment, we have the usage of the path type in flask,
        and the presence of the defensive code expected in one file. When these expectations get violated, we will lean on our game analysis
        to re-write the code, i.e., finx, if the game solution is above some threshold. This "some threshold" will be the probability you will
        see a few lines below. It will be an exciting contribution to make. Basically, for a Bayesian game, the idea we are trying to create
        is that because of the law of conservation of mass/energy, the total payoffs in our universe, which we deem as a closed system, to be
        constant. So, eventually, we can propose that everything is a constant sum game? Either way, for our base calculation, the probability
        value will derive from sources listed, which will provide the number of times certain vulnerability categories have manifested. We will
        align our example scenarios with the most common categories, input validation mainly expected, and employ a probabilistic approach, 
        which gels well with what Bayesian games need by nature.
        """
        shortest_path_2 = nx.dijkstra_path(my_graph_2, 1, 6)
        shortest_path_distance_2 = nx.shortest_path_length(my_graph_2, 1, 6, 'weight', 'dijkstra')
        print("The shortest path between 1 and 6, without EF (56), is",shortest_path_2)
        print("The distance travelled in the shortest path, without EF (56), is",shortest_path_distance_2)
        target_edge_weight = my_graph.get_edge_data(5, 6)['weight']
        print("Confirming that the weight of the edge EF (56) is",target_edge_weight)
        r"""The below formula, from the VCG lecture 3.3, is an easier, and different way to understand the main
        VCG formula for payment to an agent. Here, basically, we find the cost on others when the edge owner
        plays the game, and subtract it with the cost on others when the edge owner doesn't play the game. This
        difference, which is a measure of the costs on others with or without the target player, is the payment
        to/from the player. Payment is made to the player, if the result is negative, and the player pays, when the
        result is positive - standard VCG convention applies.
        """
        payment_ef = ((-shortest_path_distance_2) - (-shortest_path_distance+target_edge_weight))*(-1)
        print("payment to EF (56) is",payment_ef)
        profit_ef = payment_ef-target_edge_weight
        print("profit to EF (56) is",profit_ef)
        #VCG example end
        
        start_nodes_graph_1 = [n for n, d in my_graph.in_degree() if d == 0]
        end_nodes_graph_1 = [n for n, d in my_graph.out_degree() if d == 0]

        print("Start node(s) in graph 1:", start_nodes_graph_1)
        print("End node(s) in graph 1:", end_nodes_graph_1)
        
        start_nodes_graph_2 = [n for n, d in my_graph_2.in_degree() if d == 0]
        end_nodes_graph_2 = [n for n, d in my_graph_2.out_degree() if d == 0]

        print("Start node(s) in graph 2:", start_nodes_graph_2)
        print("End node(s) in graph 2:", end_nodes_graph_2)
        
        start_nodes_global_graph = [n for n, d in my_global_graph.in_degree() if d == 0]
        end_nodes_global_graph = [n for n, d in my_global_graph.out_degree() if d == 0]

        print("Start node(s) in my global graph:", start_nodes_global_graph)
        print("End node(s) in my global graph:", end_nodes_global_graph)
        
        #simrank_overall = nx.simrank_similarity(my_global_graph, None, None, 0.9, 1000, 0.0001)
        #print("The simrank in my overall graph is",simrank_overall)
        try:
            avg_shortest_path_length = nx.average_shortest_path_length(my_global_graph_undirected, None, None)
        except:
            avg_shortest_path_length = 0
        print("The average shortest path length in my_global_graph_undirected is",avg_shortest_path_length)
        num_nodes_global_undirected_graph = my_global_graph_undirected.number_of_nodes()
        print("The number of nodes in the undirected graph",num_nodes_global_undirected_graph)
        my_in_degree_iter = my_global_graph_secure.in_degree()
        my_out_degree_iter = my_global_graph_for_inout.out_degree()
        
        total_in = 0
        total_out = 0
        for node_in_degree in my_in_degree_iter:
            total_in = total_in+node_in_degree[1]
            if node_in_degree[1]>0:
                print(node_in_degree[0],"has in degree greater than 0, which is equal to",node_in_degree[1])
            
        for node_out_degree in my_out_degree_iter:
            total_out = total_out+node_out_degree[1]
            if node_out_degree[1]>0:
                print(node_out_degree[0],"has out degree greater than 0, which is equal to",node_out_degree[1])
        
        print("Total in flows",total_in)
        print("Total out flows",total_out)
        
        total_design_flaw = total_out-total_in
        
        usepath_nodes = []
        non_usepath_nodes = []
        #The custom attribute 'type' needs to be retrieved, as that is what we have defined to keep track of whether a file is using the required security library (commonvalidator) or no. If yes, then the type of the node will be commonvalidator, else no.
        for (p1, d1) in my_global_graph.nodes(data="type"):
            if d1 == 'notcommonvalidator':
                non_usepath_nodes.append(p1)
            else:
                print("Unexpected error while processing the node",p1,", which is of type",d1)
                print("Skipping this node")
                continue
        for (p2, d2) in my_global_graph_secure.nodes(data="type"):
            if d2 == 'commonvalidator':
                print("This node is using usepath as per security needs - ",p2)
                usepath_nodes.append(p2)
            else:
                print("Unexpected error while processing the node",p2,", which is of type",d2)
                print("Skipping this node")
                continue
        num_usepath_not_used = len(non_usepath_nodes)
        num_usepath_used = len(usepath_nodes)
        total_nodes = num_usepath_not_used+num_usepath_used
        print("The number of files using commonvalidator is",num_usepath_used)
        print("The number of files not using commonvalidator, i.e., the repetition of the same logic in commonvalidator, is",num_usepath_not_used)
        print("Usepath usage = ",num_usepath_used)
        print("Nodes using usepath",usepath_nodes)
        print("Nodes not using usepath",non_usepath_nodes)
        for node in non_usepath_nodes:
            print(node, "is not calling commonvalidator.py, but is simply replicating the logic. Remove it from here and replace with a call to commonvalidator.py.")

        r"""
        Logic behind the different measurements:
        1. in v. out metrics - for the design issues
        2. Avg. path - for XSS issues+design issues (actually for all categories?)
        3. Relations based metric(which?) - for SQLi. The or 1=1 is a flaw in the way the propositional logic is evaluated. 
        Irrespective of the fix techniques elsewhere, the core issue of the logic abuse has never been fixed. Measuring this can be a good starting point.
        Cyclical graph for tautology? Modeling relations as graphs, and analyzing the properties?
        
        Avg path calculation could be the low level metric, and cat. specific metric the high level. So, avg path could apply to call cats.
        For XSS, the high level metric could be language level, where 3 is good (regex/strict). For SQL, relation metric
        
        The avg path is to prevent repetition of the high level defensive measure.
        
        f(avg path)_supergame = sigma_cat f(avg path length)_cat
        OR
        f(metric)_supergame = sigma_cat f(metric)_cat
        OR
        f(metric)_universalgame = sigma_cat f(metric)_cat
        
        in our case
        f(metric)_universalgame = f(in-out-metric)_design_issue + f(avg_path_length)_input_validation + f(tautology_outcome)_SQLi_tautology + f(yet_to_be_realized_games_payoffs)
        tautology metrics - consistency? equiconsistency? soundness? validity?
        
        f(whatever) = operations on matrices? Representation of matrices?
        
        Another thing to do: Make the Spaniel-utility above for n matrices instead of the 2 shown above.
        
        Argument to make - treat security overall (for all categories) as an isolated system, and if any leak, play the game to handle it within
        the other isolated system we are in? or assume from start that security isn't isolated, and compare with functionality etc. The
        "leaking" quantity here is STEM related (let's say gap in logic of one vuln. that doesn't fit into some functionality, and 
        not money or other economic ones. So, each time we assume isolated system, but that isn't the case, we add a matrix. This 
        might be a bit problematic on the time complexity, so would be good to comment on it.
        
        """
        tautology_cycles_without = 1 #to-do: make a call to the db file and get the number from there, or find a way to have it run through here.
        tautology_cycles_with = 0 #to-do: make a call to the db file and get the number from there, or find a way to have it run through here.
        math_compatibility_factor = 1 #between 0 and 1?
        math_incompatibility_factor = 1 #between 0 and 1?
        sdlc_cycles = 1 #T.B.C: number of branches/commits can indicate cyclicity factor?
        #need to add the VCG style formula implementation of with/without the player below, to justify things
        #Simplified VCG formula for coding the VCG function above:
        #Paymet_to_target_node = ((cost to others without target node)-(cost to others with the target node))*(-1)
        # OR
        #Payment_to_target_node = ((-category_metric_without_target_player)-(-category_metric_with_target_player))*(-1)
        payment_to_others = (num_usepath_not_used+avg_shortest_path_length+tautology_cycles_without)/3
        payment_to_others = round(payment_to_others*math_compatibility_factor*math_incompatibility_factor*sdlc_cycles,2)
        payment_to_others_design = num_usepath_not_used
        payment_to_others_design = round(payment_to_others_design*math_compatibility_factor*math_incompatibility_factor*sdlc_cycles,2)
        payment_to_others_iv = avg_shortest_path_length
        payment_to_others_iv = round(payment_to_others_iv*math_compatibility_factor*math_incompatibility_factor*sdlc_cycles,2)
        payment_to_others_sqli = tautology_cycles_without
        payment_to_others_sqli = round(payment_to_others_sqli*math_compatibility_factor*math_incompatibility_factor*sdlc_cycles,2)
        #VCG payoff formula in action - the inner negative numbers are to express the values as an expense and not income. And the multiplication by -1 is to express
        #the convention of VCG (where the player who improves the game for all, will result in a negative VCG output)
        # in the final value that come out as per the previous paper. The payoff matrix value is the VCG value times -1, based on this convention.
        design_complexity_without = num_files_processed
        design_complexity_with = num_files_processed-total_in
        #payment_to_sec_design = ((-design_complexity_without)-(-(design_complexity_with)))*(-1)
        payment_to_sec_design = get_payoffs_from_vcg(design_complexity_without,design_complexity_with)
        payment_to_sec_design = round(payment_to_sec_design*math_compatibility_factor*math_incompatibility_factor*sdlc_cycles,2)
        iv_complexity_without = num_nodes_global_undirected_graph
        iv_complexity_with = avg_shortest_path_length
        #payment_to_sec_iv = ((-iv_complexity_without)-(-iv_complexity_with))*(-1)
        payment_to_sec_iv = get_payoffs_from_vcg(iv_complexity_without,iv_complexity_with)
        payment_to_sec_iv = round(payment_to_sec_iv*math_compatibility_factor*math_incompatibility_factor*sdlc_cycles,2)
        #payment_to_sec_sqli = ((-tautology_cycles_without)-(-tautology_cycles_with))*(-1)
        payment_to_sec_sqli = get_payoffs_from_vcg(tautology_cycles_without, tautology_cycles_with)
        payment_to_sec_sqli = round(payment_to_sec_sqli*math_compatibility_factor*math_incompatibility_factor*sdlc_cycles,2)
        
        print("Payment to others", payment_to_others)
        print("Payment to sec for securing design vulnerabilities",payment_to_sec_design)
        print("Payment to sec for securing input validation vulnerabilities",payment_to_sec_iv)
        print("Payment to sec for securing SQL injection vulnerabilities",payment_to_sec_sqli)
        
        p = 0.24#From OWASP for occurrence rate of design issues: https://owasp.org/Top10/A03_2021-Injection/
        #For the probability, we will use the idea of conservation of wealth via material/energy types of wealth
        #Think of this probability as the "guide" on the risk/reward of vuln. categories.
        #Play around with the probability values and provide various Game Outputs?! Play around for combined Bayesian game, and separate NF games?
        #Where is the conservation/symmetry/compatibility idea most relevant? VCG calculation? Probability calculation? Conservation for risk management and intangible representation, symmetry for design/IV
        #The story to be told here is the merit of combining across categories, and reasoning out which ones to finx/which not (i.e. the dev to fix). If added to defect backlog, then increase penalty.
        p_des = 0.24
        p_sqli = 0.15#avg of OWASP Injection and CVE
        p_iv = 0.28#iv+xss, see if it needs to be averaged with the OWASP percentage
        r"""
        ---------------Meeting 30-08-2023---------------
    
            Get the probabilities
                from law of conservation? The idea of risk management can be quantified using this approach.
                    where risk = math_compatibility_factor*math_incompatibility_factor?
                Nash Bargaining Solution?
                Size of automata/other automata metrics for input validation?
                    Any other STEM metric that is equivalent to input validation or other vuln categories?
                https://www.cvedetails.com/vulnerabilities-by-types.php
                    total-all-cats-10-years=108,141
                    total-sqli-10-years from 2015 to 2025=12318/108,141 (0.1139)
                    total-iv-10-years=5840/108,141 (0.054)
                    total-xss=34058 (0.3149)
                    total-design from OWASP = (0.2419)
                    total-injection from OWASP = (0.1909)
                Get the dollar value per category from OWASP (and any other source too)
                CWE link?
                https://blog.cloudflare.com/application-security-report-q2-2023/
                https://www.indusface.com/research-reports/state-of-appsec-report-q1-2023.pdf?utm_medium=email&_hsmi=258835016&_hsenc=p2ANqtz--XgW29dM0wuwo-_PwVPzLdjZHDgNuP0uHB1re6MQwOd1tGhEldWS-ZSX5rhhmL9jIGLsxGXYw8q-rYL5hLlUkmdkW6fepoXb1_Nr_pDZ7N8Yj10_U&utm_content=258835016&utm_source=hs_automation
                https://www.datadoghq.com/state-of-application-security/
            Get also a threshold based on the above/new criteria
                CVE/CVSS to above?
                Industry specific? Low v high risk appetite for military (life/physical asset loss) v social media (non-physical loss)?
            Compute the automata metrics for input validation
            
            3/10/2023
            FOR THE CODE
                probab values above
                Get graph/data structure from SAST tool
                    VCG
                    pybandit
                    ??
                    This will give us the payoff matrix, which will solve the game
                    is this a graph v automata, or graph to automata type game to check out?
                """
        
        """
        Example from Spaniel:
        PD: 0.2
                L|R
        Up --> 3,3|0,4
        Down --> 2,1|1,2
        
        ONE-STONE-MULTIPLE-TARGETS
        XSS/Input Validation/Design Issue (multiply by line numbers, and release cycle count pending)
        Regular Payoff Matrix from Jagged  (R_iv)                  Jagged Payoff Matrix (J_iv) (idea derived from Irregular Matrix/Jagged Arrays)
                Finx|Don't                                                    Finx|Don't
        Fix    -23,20|-20,-20                                      Fix        -23,20|-20,-20
                                                
        Don't  0,20|unknown,unknown                                Don't        0,20|
        
        SH: 0.8
                L|R
        Up --> 3,3|0,2
        Down --> 2,0|1,1
        
        ONE-STONE-ONE-TARGET
        SQL Injection Tautology based (multiply, for a DB vendor, by every user in the world who has to parameterize the query)
        Regular Payoff Matrix from Jagged  (R_db)                     Jagged Payoff Matrix (J_db)
                Finx|Don't                                                     Finx|Don't
        Fix     -23,3|-3,-3                                           Fix      -23,3|-3,-3
        
        Don't   0,3|unknown,unknown                                   Don't     0,3|
        
        The combined matrix (we could argue to re-composing matrices to the universal game here):
                        LL|LR|RL|RR
        Up -->        3,3|0.6,2.2|2.4,3.2|0,2.4
        Down -->    2,0.2|1.2,1|1.8,0.4|1,1.2

        Converted to our game:
        Design/XSS: 0.x - probabilities based on CVE data
                Finx|Don't
        Fix    a,b|c,d
        Don't  e,f|g,h
        a to h values = VCG(returns in-out metrics/avg. path length) based on graph outputs of the analyzed pathrule code
        SQLi: 0.y - probabilities based on CVE data
                Finx/Don't
        Fix    i,j|k,l
        Don't  m,n|o,p
        i to p values = VCG(returns database-graph property such as cyclic graph due to tautology) based on graph outputs of the analyzed db code
        Simple rule - if cycle found, then tautology, hence finx. No questions asked for this category, as it's the DB vendor code (like mysql) 
        Can be less strict for above.
        
        Above, we went from jagged to regular payoff matrix by stating "unknown" in the cell locations that were empty in the 
        jagged payoff matrix
        
        Question: for now, convert jagged matrix into regular matrix by saying value = 0 in missing places? But is that a true indication
        of the value? What if it's negative, or even positive? We could introduce a residue/error, similar to the constant of
        integration, and then proceed. Let's also implement Iliffe vectors, if in doubt. Or maybe, add a placeholder in the missing cells,
        and let those placeholders indicate unknowns, to not be acted upon.
        
        1. Expect nulls, catch it, and do something?
        2. Something = assume it 0 for the time being, average this and other payoffs, and populate some other list for this assumption? The list being a warning, saying, assuming 0 for now, but unsure if actual payoff is a bad negative value?
        3. Something = something else? Like -(OWASP estimate/some other estimate of loss) 
        4. Something = leave it for non-technical/financial/risk function to solve?
        5. Something = set worst case value of a hack to be = num of ALL files owned by company*num of lines (simplistic monetary value of code from a dev perspective, to showcase the worst case being a company shutting down, which makes ALL their code useless, and that's the loss from a dev perspective)
        
        Future work: improve game analysis techniques for games with jagged payoff matrices.
        
        """
        #in addition to the graph with in-out metrics, is another graph possible to visualize for the same category with finx. Should those 2 be compared with those of other categories in the Bayesian analysis? Or do we keep them separate as analyses? 
        u1 = np.array([[-(payment_to_others),-(payment_to_others)], [0,"unknown"]]) #To be defined by calling game_helper.add_games_payoffs or multiply_games_payoffs, recompose a game
        U1 = [u1, u1]
        u1_design = np.array([[-(payment_to_others_design),-(payment_to_others_design)], [0,-10000]])
        u1_iv = np.array([[-(payment_to_others_iv),-(payment_to_others_iv)], [0,-10000]])
        u1_sqli = np.array([[-(payment_to_others_sqli),-(payment_to_others_sqli)], [0,-10000]])
        A1 = ['U', 'D']
        
        u21 = np.array([[payment_to_sec_design,0], [payment_to_sec_design,"unknown"]]) #To be defined by calling game_helper.add_games_payoffs or multiply_games_payoffs, recompose a game
        u2_design = np.array([[payment_to_sec_design*p_des,0], [payment_to_sec_design*p_des,-10000]])
        u22 = np.array([[payment_to_sec_iv,0], [payment_to_sec_iv,"unknown"]]) #randomly made by me, need refining, but more to capture that sec may not finx all - i.e 20 instead of 21 hops, maybe due to false positive/other reason#To be defined by calling game_helper.add_games_payoffs or multiply_games_payoffs, recompose a game
        u2_iv = np.array([[payment_to_sec_iv*p_iv,0], [payment_to_sec_iv*p_iv,-10000]])
        U2 = [u21, u22]
        u23 = np.array([[payment_to_sec_sqli,0], [payment_to_sec_sqli,"unknown"]])
        u2_sqli = np.array([[payment_to_sec_sqli*p_sqli,0], [payment_to_sec_sqli*p_sqli,-10000]])
        U3 = [u21, u23]
        U21 = [u21,u21]
        U22 = [u22,u22]
        a2 = ['L', 'R']
        A2 = [f'{a}{b}' for a in a2 for b in a2]
        #types for categories? Or was it for finx/find only?
        print(f'---- Start: If P2 (security) is type 0, the payoffs are -----')
        tab1 = print_payoffs([u1, u21], [A1, a2])
        print(tab1)
        print(f'---- End: If P2 (security) is type 0, the payoffs are -----')
        
        print(f'---- Start: If P2 is type 1, the payoffs are -----')
        tab2 = print_payoffs([u1, u22], [A1, a2])
        print(tab2)
        print(f'---- End: If P2 is type 1, the payoffs are -----')
        
        print(f'----- Start: Compute full matrix of all types -------')
        t1, t2, A1, A2 = compute_full_matrix(U1, U2, p, [A1, a2])
        tab_combined = print_payoffs([t1, t2], [A1, A2], 3)
        print(tab_combined)
        #comment/uncomment combined sqli game for paper/thesis as required
        t3, t4, A3, A4 = compute_full_matrix(U1, U3, p, [A1, a2])
        tab_combined_sql = print_payoffs([t3, t4], [A3, A4], 3)
        print(f'----- End: Compute full matrix of all types -------')
        
        print(f'----- Start: u1 -------')
        print(u1)
        print(f'----- End: u1 -------')
        
        print(f'----- Start: u21 -------')
        print(u21)
        print(f'----- End: u21 -------')
        
        print(f'----- Start: u22 -------')
        print(u22)
        print(f'----- End: u22 -------')
        
        print(f'----- Start: tab1 -------')
        print(tab1)
        print(f'----- End: tab1 -------')
        
        print(f'----- Start: tab2 -------')
        print(tab2)
        print(f'----- End: tab2 -------')
        
        print(f'----- Start: tab_combined -------')
        print(tab_combined)
        print(f'----- End: tab_combined -------')
        #comment/uncomment combined sqli game for paper/thesis as required
        print(f'----- Start: tab_combined_sql -------')
        print(tab_combined_sql)
        print(f'----- End: tab_combined_sql -------')
        print(f'----- Start: A_ --------')
        print("Sending the following t1 and t2 to the IESDS method")
        print("t1")
        print(t1)
        print("t2")
        print(t2)
        #comment/uncomment combined sqli game for paper/thesis as required
        print("t3")
        print(t3)
        print("t4")
        print(t4)
        A_, T_ = IESDS([A1, A2], [t1, t2], DOPRINT=True)
        strategy_security = A_[1]
        print(type(strategy_security))
        print("The security strategy for the combined game")
        #comment/uncomment combined sqli game for paper/thesis as required
        A2_, T2_ = IESDS([A3, A4], [t3, t4], DOPRINT=True)
        strategy_security_sql = A_[1]
        print(type(strategy_security_sql))
        print(A_[1])
        print("overall A_")
        print(A_)
        print(f'----- End: A_ --------')
        print(f'----- Start: T_ --------')
        print(T_)
        print(len(T_))
        print(type(T_))
        print(T_[0][0][0])
        print(T_[1][0][0])
        print(f'----- End: T_ --------')
        
        """
        #Not needed anymore these section, especially since it's erroring out after adding support for unknowns. N.A. for the PhD.
        print(f'----- Start: IESDS from method --------')
        #tab_iesds = print_payoffs(T_, A_, 3)
        print(f'----- End: IESDS from method --------')
        print(f'----- Start: IESDS here --------')
        #print(tab_iesds)
        print(f'----- End: IESDS here --------')
        """
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.size': 40})
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"]
        # Showing each game's metric side by side
        """
        # Sample data
        categories = ['Category A', 'Category B', 'Category C', 'Category D']
        values = [25, 40, 30, 55]
        
        # Create the bar chart
        plt.bar(categories, values)
        
        # Add labels and title
        plt.xlabel('Categories')
        plt.ylabel('Values')
        plt.title('Simple Bar Chart')
        
        # Display the chart
        plt.show()
        
        cats = ['A', 'B', 'C', 'D'] # categories
        vals1, vals2, vals3 = [4, 5, 6, 7], [3, 4, 5, 6], [2, 3, 4, 5]
        
        # Bar width and x locations
        w, x = 0.05, np.arange(len(cats))
        
        fig, ax = plt.subplots()
        ax.bar(x - w, vals1, width=w, label='Set 1')
        ax.bar(x, vals2, width=w, label='Set 2')
        ax.bar(x + w, vals3, width=w, label='Set 3')
        
        ax.set_xticks(x)
        ax.set_xticklabels(cats)
        ax.set_ylabel('Values')
        ax.set_title('Grouped Bar Chart')
        ax.legend()
        
        plt.show()
        """
        cats = ['IV', 'SQLi', 'Design'] # categories
        #vals1, vals2, vals3, vals4, vals5 = [avg_shortest_path_length, tautology_cycles_without, total_design_flaw], [avg_shortest_path_length*2, tautology_cycles_without*2, total_design_flaw*2], [avg_shortest_path_length*3, tautology_cycles_without*3, total_design_flaw*3], [avg_shortest_path_length*4, tautology_cycles_without*4, total_design_flaw*4], [avg_shortest_path_length*5, tautology_cycles_without*5, total_design_flaw*5]
        vals1, vals2, vals3 = [avg_shortest_path_length, tautology_cycles_without, total_design_flaw], [avg_shortest_path_length*2, tautology_cycles_without*2, total_design_flaw*2], [avg_shortest_path_length*3, tautology_cycles_without*3, total_design_flaw*3]
        # Bar width and x locations
        w, x = 0.05, np.arange(len(cats))
        
        fig, ax = plt.subplots()
        #ax.bar(x-(w*2), vals1, width=w, label='Set 1')
        ax.bar(x-w, vals1, width=w, label="Cycle 1")
        ax.bar(x, vals2, width=w, label="Cycle 2")
        ax.bar(x+w, vals3, width=w, label="Cycle 3")
        #ax.bar(x+(w*2), vals5, width=w, label='Set 5')
        
        ax.set_xticks(x)
        ax.set_xticklabels(cats)
        ax.set_ylabel('Values')
        #ax.set_title('Cost per category per cycle (for each category, the bars from left to right show a cycle of 1 to 5)')
        ax.legend(fontsize=24)
        
        #plt.show() #uncomment after getting timing data. This was especially done after submitting to the journal, as Code Ocean works differently. Comment/uncomment the pylab.show() as is necessary.
        
        #to:do - show following charts - each game's metric individually, and for the universal game (all 3 together)
        """
        #Support enumeration doesn't seem to be required for our design, as IESDS on the combined matrix works better
        eqs = list(nashpy.Game(T_[0], T_[1]).support_enumeration())
        print(f'Found {len(eqs)} equilibria')
        print(eqs)
        print(type(eqs))
        for i,eq in enumerate(eqs):
            print(f'{i+1}: s1 = {eq[0]}, s2 = {eq[1]}')
        """
        nashpy_game_design = nashpy.Game(u1_design,u2_design)
        print("nashpy_game_design")
        print(nashpy_game_design)
        eqs_design = nashpy_game_design.support_enumeration()
        print("nashpy eqs_design")
        print(list(eqs_design))
        sigma_dev = np.array([0, 1])#pre-calculated eqs and then hardcoded. Need an automated way of reading eqs_* and building this array from it
        sigma_sec = np.array([1, 0])
        design_game_solution = nashpy_game_design[sigma_dev, sigma_sec]
        print("Design Game Solution Utilities/Payoffs")
        print(design_game_solution)
        nashpy_game_iv = nashpy.Game(u1_iv,u2_iv)
        print("nashpy_game_iv")
        print(nashpy_game_iv)
        eqs_iv = nashpy_game_iv.support_enumeration()
        print("nashpy eqs_iv")
        print(list(eqs_iv))
        iv_game_solution = nashpy_game_iv[sigma_dev, sigma_sec]
        print("IV Game Solution Utilities/Payoffs")
        print(iv_game_solution)
        nashpy_game_sqli = nashpy.Game(u1_sqli,u2_sqli)
        print("nashpy_game_sqli")
        print(nashpy_game_sqli)
        eqs_sqli = nashpy_game_sqli.support_enumeration()
        print("nashpy eqs_sqli")
        print(list(eqs_sqli))
        sqli_game_solution = nashpy_game_sqli[sigma_dev, sigma_sec]
        print("SQLi Game Solution Utilities/Payoffs")
        print(sqli_game_solution)
        """
        # 2. Generate x-values for the sample convex function. Obviously, please change this to the convex game
        x_values = np.linspace(-5, 5, 100) # 100 points between -5 and 5
        
        # 3. Calculate corresponding y-values. Obviously, please change this to the convex game
        y_values = convex_function(x_values)
        
        # 4. Plot the function
        plt.figure(figsize=(8, 6))
        plt.plot(x_values, y_values, label=r'$f(x) = x^2$', color='blue')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Plot of a Convex Function')
        plt.grid(True)
        plt.legend()
        plt.show()
        """
        x_values = [0.00,20.25,0.00]
        y_values = [1,2,1]
        # Plot the points as circles
        plt.plot(x_values, y_values, 'o-')
        
        # Add labels and title (optional)
        plt.xlabel("v[S]")
        plt.ylabel("n")
        #plt.title("Convexity of the core - IV and Design combined game")
        plt.tight_layout()
        # Display the plot
        #plt.show() #uncomment after getting timing data. This was especially done after submitting to the journal, as Code Ocean works differently. Comment/uncomment the pylab.show() as is necessary.
        
        x_values = [0.00,1.00,0.00]
        y_values = [1,2,1]
        # Plot the points as circles
        plt.plot(x_values, y_values, 'o-')
        
        # Add labels and title (optional)
        plt.xlabel("v[S]")
        plt.ylabel("Sequence number")
        #plt.title("Convexity of the core - SQLi and Design combined game")
        
        # Display the plot
        #plt.show() #uncomment after getting timing data. This was especially done after submitting to the journal, as Code Ocean works differently. Comment/uncomment the pylab.show() as is necessary.
        
        x_values = [0.00,0.24,7.37,20.25,0.00]
        y_values = [1,2,3,4,1]
        # Plot the points as circles
        plt.plot(x_values, y_values, 'o-')
        
        # Add labels and title (optional)
        plt.xlabel("v[S]")
        plt.ylabel("Sequence number")
        #plt.title("IV and Design games payoffs (separate and combined)")
        
        # Display the plot
        #plt.show() #uncomment after getting timing data. This was especially done after submitting to the journal, as Code Ocean works differently. Comment/uncomment the pylab.show() as is necessary.
        x_values = [0.00,0.15,0.24,1.00,0.00] #Individual payoffs of SQLi and Design games, and Combined one, for Security player
        y_values = [1,2,3,4,1]
        # Plot the points as circles
        plt.plot(x_values, y_values, 'o-')
        
        # Add labels and title (optional)
        plt.xlabel("v[S]")
        plt.ylabel("Sequence number")
        #plt.title("Convexity of the payoffs for the security player - SQLi and Design games, separate and combined")
        
        # Display the plot
        #plt.show() #uncomment after getting timing data. This was especially done after submitting to the journal, as Code Ocean works differently. Comment/uncomment the pylab.show() as is necessary.
        for strategy_string in strategy_security:
            for strategy in strategy_string:
                if strategy == 'L':
                    #Use modularity/cyclicity for decision?
                    print("will finx this file/line, if the annotation @finx is encountered")
                else:
                    print("Don't finx")
        
        #comment/uncomment combined sqli game for paper/thesis as required
        for strategy_string_sql in strategy_security_sql:
            for strategy_sql in strategy_string_sql:
                if strategy_sql == 'L':
                    #Use modularity/cyclicity for decision?
                    print("will finx this file/line for sql, if the annotation @finx is encountered")
                else:
                    print("Don't finx")
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print("Elapsed time is",elapsed_time)
        #FINAL NUMBERS TO SHOW/INTERPRET - speed/accuracy/something else? - work with experimental values
        #SHOW THEORETICAL STRENGTH FOR RECOMPOSING TO UNIVERSAL GAME? - work with dummy values
        #ARGUE ABOUT MIX OF (THEORETICAL STRENGTH+EXPERIMENTAL STRENGTH)/2?
        #T.B.C if some condition as per above is met (derived from conservation principle and the probability above), then re-write the source, and check-in
        #T.B.C, if A_ for security is L, then finx
        # Unparse e.g.: https://stackoverflow.com/questions/3774162/given-an-ast-is-there-a-working-library-for-getting-the-source
        # Modify AST and write back source: https://stackoverflow.com/questions/768634/parse-a-py-file-read-the-ast-modify-it-then-write-back-the-modified-source-c
        #T.B.C If not re-write source, then aspects?
        #repetition/secure code in wrong location is decomposed game, we then combine/re-compose it
        #for file in non_commonvalidator_nodes:
            #print(bandit.linecache.getlines(file,module_globals=None))
    else:
        print("we have not processed all the expected files")
