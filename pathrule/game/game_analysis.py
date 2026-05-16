"""
TBC //Bayesian game analysis here (LCP implementation). The various payoffs, for the different vuln. Categories, and the split of the call graphs, will equate to various games for the Bayesian Setting
// if cooperation is not the outcome, will predict that an unknown player is making it a zero sum game, which will be a NEW CONCEPT in the literature. We can then use LP here, instead of LCP
return outcome_from_lcp/lp_solver // will use existing LCP/LP solver
"""
import bimatrix as bm
import pandas as pd
import numpy as np
import itertools
import nashpy

def compute_full_matrix(U1, U2, p, action_names=None):
    """      
    "        Source of this method: https://github.com/GamEconCph/2023-lectures/blob/main/Bayesian%20Games/BNE.ipynb?short_path=6214124
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
    t1 = np.empty((nA1, nA2*nA2))
    t2 = np.empty((nA1, nA2*nA2))
    
    for ia1 in range(nA1):
        i_col = 0
        
        for a2_1 in range(nA2):
            for a2_2 in range(nA2):
                t1[ia1,i_col] = p*U1[0][ia1,a2_1] + (1.-p)*U1[1][ia1,a2_2]
                t2[ia1,i_col] = p*U2[0][ia1,a2_1] + (1.-p)*U2[1][ia1,a2_2]
                
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
        

#def game_analysis(payoffs):

def main():
    p = 0.2#To be defined by calling game_helper.get_bayesian_game_probability, from standard datasets provided by authorities+vendors
    #For the probability, we will use the idea of conservation of wealth via material/energy types of wealth
    
    """
    ---------------Meeting 30-08-2023---------------

Get the probabilities
    from law of conservation?
    Nash Bargaining Solution?
    Size of automata/other automata metrics for input validation?
        Any other STEM metric that is equivalent to input validation or other vuln categories?
    https://www.cvedetails.com/vulnerabilities-by-types.php
        total-all-cats-10-years=83557
        total-sqli-10-years=6877 (0.0823)
        total-iv-10-years=8096 (0.0969)
        total-xss=20982 (0.2511)
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
    
    u1 = np.array([[3,0], [2,1]]) #To be defined by calling game_helper.add_games_payoffs or multiply_games_payoffs, recompose a game
    U1 = [u1, u1]
    A1 = ['U', 'D']
    
    u21 = np.array([[3,4], [1,2]]) #To be defined by calling game_helper.add_games_payoffs or multiply_games_payoffs, recompose a game
    u22 = np.array([[3,2], [0,1]]) #To be defined by calling game_helper.add_games_payoffs or multiply_games_payoffs, recompose a game
    U2 = [u21, u22]
    a2 = ['L', 'R']
    A2 = [f'{a}{b}' for a in a2 for b in a2]
    
    print(f'---- If P2 is type 0 -----')
    bm.print_payoffs([u1, u21], [A1, a2])
    
    print(f'---- If P2 is type 1 -----')
    bm.print_payoffs([u1, u22], [A1, a2])
    
    t1, t2, A1, A2 = compute_full_matrix(U1, U2, p, [A1, a2])
    
    bm.print_payoffs([t1, t2], [A1, A2], 3)
    
    A_, T_ = bm.IESDS([A1, A2], [t1, t2], DOPRINT=True)
    
    bm.print_payoffs(T_, A_, 3)
    
    eqs = list(nashpy.Game(T_[0], T_[1]).support_enumeration())
    print(f'Found {len(eqs)} equilibria')
    for i,eq in enumerate(eqs):
        print(f'{i+1}: s1 = {eq[0]}, s2 = {eq[1]}')
    
    
if __name__ == '__main__':
    main()
    
    
