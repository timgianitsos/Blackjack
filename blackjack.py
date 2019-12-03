import itertools
import matplotlib.pyplot as plt
import numpy as np
import collections
import operator as op
from functools import reduce

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer / denom

def action_space(): return [0,1]

def state_space(max_card_val):
    all_deck = list(itertools.product([0,1, 2, 3, 4], repeat=max_card_val))

#state of 1 dealer and 1 player = (cards in deck, cards in player's hand, cards in dealer's hand)

def calc_handscore(hand):
    score = 0
    for card in range(1,len(hand)):
        score += (card+1)*hand[card]
    #hand[0] special case for aces
    ace_value = list(itertools.product([1,11],repeat=hand[0]))
    max_score = score
    for entry in ace_value:
        possible_comb_ace = list(entry)
        possible_comb_ace.append(score)
        local_score = sum(possible_comb_ace)
        if local_score > max_score:
            max_score = local_score
    return max_score

def naive_policy(state):
    if calc_handscore(state[1]) < 16:
        return 1
    else:
        return 0

def draw_card_from_deck(deck):
    possible_cards = []
    for card in range(0,len(deck)):
        if deck[card] > 0:
            for i in range(0,deck[card]):
                possible_cards.append(card)
    if len(possible_cards) == 0:
        return -1
    else:
        drawn_card = possible_cards[np.random.randint(0,len(possible_cards))]
        return drawn_card


def compare_hand(player,dealer):
    playscore = calc_handscore(player)
    dealscore = calc_handscore(dealer)
    if playscore > 21:
        return -1
    if dealscore > 21:
        return 1
    if playscore > dealscore:
        return 1
    elif playscore == dealscore:
        return 0
    else:
        return -1

def generate_new_game(state):
    print("generating new game!")
    new_state = state
    state[1] = [0 for i in range(0,len(state[1]))]
    state[2] = [0 for i in range(0,len(state[2]))]
    for player in range(0,(len(state)-1)*2):
        drawn_card = draw_card_from_deck(new_state[0])
        if drawn_card == -1:
            #no more cards in deck
            return ([[0 for i in range(0,len(state[0]))]for j in range(0,len(state))],-1)
        new_state[0][drawn_card] += -1
        new_state[player%2+1][drawn_card] += 1
    return new_state

def generate_nstate_reward(state,action):
    if action == 1:
        drawn_card = draw_card_from_deck(state[0])
        if drawn_card == -1:
            #no more cards in deck
            return ([[0 for i in range(0,len(state[0]))]for j in range(0,len(state))],compare_hand(state[1],state[2]))
        new_state = state
        new_state[0][drawn_card] += -1
        new_state[1][drawn_card] += 1
        handscore = calc_handscore(new_state[1])
        if handscore > 21:
            #bust! generate another game
            print("agent chose to hit and busted!",calc_handscore(new_state[1])," dealer=",calc_handscore(new_state[2]))
            new_state = generate_new_game(new_state)
            return (new_state,-1)
        elif handscore == 21:
            #blackjack!
            print("agent chose to hit and got blackjack!",calc_handscore(new_state[1])," dealer=",calc_handscore(new_state[2]))
            new_state = generate_new_game(new_state)
            return (new_state,1)
        else:
            print("agent chose to hit! agent=",calc_handscore(new_state[1])," dealer=",calc_handscore(new_state[2]))
            return (new_state,0)
    else:
        #action == 0
        new_state = state
        while calc_handscore(new_state[2]) < 17:
            drawn_card = draw_card_from_deck(new_state[0])
            new_state[0][drawn_card] += -1
            new_state[2][drawn_card] += 1
        reward = compare_hand(new_state[1],new_state[2])
        if reward == 1:
            print("agent chose to stay and won!",calc_handscore(new_state[1])," dealer=",calc_handscore(new_state[2]))
        elif reward == -1:
            print("agent chose to stay and lost!",calc_handscore(new_state[1])," dealer=",calc_handscore(new_state[2]))
        else:
            print("agent chose to stay and tied!",calc_handscore(new_state[1])," dealer=",calc_handscore(new_state[2]))
        #generate new game
        generate_new_game(new_state)
        return (new_state,reward)

def rollout(state,depth,policy_func):
    if depth == 0:
        return 0
    action = policy_func(state)
    next_state,reward = generate_nstate_reward(state,action)
    return reward + rollout(next_state,depth-1,policy_func)

def initial_state():
    deck = [4 for i in range(0,10)]
    hand = [0 for i in range(0,10)]
    return [deck,hand.copy(),hand.copy()]

class MCTS:
    def __init__(self):
        self.h = {}
        self.T = set()
        self.N_h = collections.defaultdict(int)
        self.N_ha = collections.defaultdict(int)
        self.Q = collections.defaultdict(float)
        self.N_ha_nought = {}
        self.Q_nought = {}
        self.c = 1.
        self.gamma = 1.

def generate_new_game_withobserv(state):
    print("generating new game!")
    observation = []
    new_state = state
    state[1] = [0 for i in range(0,len(state[1]))]
    state[2] = [0 for i in range(0,len(state[2]))]
    for player in range(0,(len(state)-1)*2):
        drawn_card = draw_card_from_deck(new_state[0])
        observation.append(drawn_card)
        if drawn_card == -1:
            #no more cards in deck
            return (([[0 for i in range(0,len(state[0]))]for j in range(0,len(state))],-1),observation)
        new_state[0][drawn_card] += -1
        new_state[player%2+1][drawn_card] += 1
    return (new_state,observation)

#(s',o,r) ~ G(s,a)
def generate_nstate_observ_reward(state,action):
    observation = []
    if action == 1:
        drawn_card = draw_card_from_deck(state[0])
        observation.append(drawn_card)
        if drawn_card == -1:
            #no more cards in deck
            return (([[0 for i in range(0,len(state[0]))]for j in range(0,len(state))],compare_hand(state[1],state[2])),observation)
        new_state = state
        new_state[0][drawn_card] += -1
        new_state[1][drawn_card] += 1
        handscore = calc_handscore(new_state[1])
        if handscore > 21:
            #bust! generate another game
            print("agent chose to hit and busted!",calc_handscore(new_state[1])," dealer=",calc_handscore(new_state[2]))
            new_state,newobserv = generate_new_game_withobserv(new_state)
            observation.extend(newobserv)
            return (new_state,tuple(observation),-1)
        elif handscore == 21:
            #blackjack!
            print("agent chose to hit and got blackjack!",calc_handscore(new_state[1])," dealer=",calc_handscore(new_state[2]))
            new_state,newobserv = generate_new_game_withobserv(new_state)
            observation.extend(newobserv)
            return (new_state,tuple(observation),1)
        else:
            print("agent chose to hit! agent=",calc_handscore(new_state[1])," dealer=",calc_handscore(new_state[2]))
            return (new_state,tuple(observation),0)
    else:
        #action == 0
        new_state = state
        while calc_handscore(new_state[2]) < 17:
            drawn_card = draw_card_from_deck(new_state[0])
            observation.append(drawn_card)
            new_state[0][drawn_card] += -1
            new_state[2][drawn_card] += 1
        reward = compare_hand(new_state[1],new_state[2])
        if reward == 1:
            print("agent chose to stay and won!",calc_handscore(new_state[1])," dealer=",calc_handscore(new_state[2]))
        elif reward == -1:
            print("agent chose to stay and lost!",calc_handscore(new_state[1])," dealer=",calc_handscore(new_state[2]))
        else:
            print("agent chose to stay and tied!",calc_handscore(new_state[1])," dealer=",calc_handscore(new_state[2]))
        #generate new game
        new_state,newobserv = generate_new_game_withobserv(new_state)
        observation.extend(newobserv)
        return (new_state,tuple(observation),reward)

def simulate(mcts,state,history,depth):
    if depth == 0:
        return 0
    if history not in mcts.T:
        for act in action_space():
            if (history,act) in mcts.N_ha_nought:
                mcts.N_ha[(history,act)] = mcts.N_ha_nought[(history,act)]
            if (history,act) in mcts.Q_nought:
                mcts.Q[(history,act)] = mcts.Q_nought[(history,act)]
        mcts.T.add(history)
        return rollout(state,depth,naive_policy) #policy dependent on belief?

    #argmax_a Q(h,a) + csqrt(logN(h)/N(h,a))
    opt_act = 0
    max_q = float('-inf')
    tmp_qval = []
    for act in action_space():
        if mcts.N_ha[(history,act)] > 0 and mcts.N_h[history] > 0:
            q_val = mcts.Q[(history,act)] + mcts.c*np.sqrt(np.log(mcts.N_h[history])/mcts.N_ha[(history,act)])
        else:
            q_val = mcts.Q[(history,act)]
        if q_val > max_q:
            max_q = q_val
            opt_act = act
        tmp_qval.append(q_val)
    if tmp_qval[0] == tmp_qval[1]:
        #no optimal action, choose random action
        opt_act = np.random.randint(0,2)

    #(s',o,r) ~ G(s,a)
    new_state,observation,reward = generate_nstate_observ_reward(state,opt_act)
    newhistory = history
    newhistory = newhistory + (opt_act,observation)
    new_q = reward + mcts.gamma* simulate(mcts,new_state,newhistory,depth-1)
    mcts.N_ha[(history,opt_act)] += 1
    mcts.Q[(history,opt_act)] += (new_q - mcts.Q[(history,opt_act)])/mcts.N_ha[(history,opt_act)]
    return new_q


def samplebelief(belief):
    return generate_new_game(initial_state())

def selectaction(mcts,belief,depth,max_iters):
    history = tuple()
    for iter in range(0,max_iters):
        sampled_state = samplebelief(belief)
        simulate(mcts,sampled_state,history,depth)

    max_q = float('-inf')
    opt_act = 0
    tmp_qval = []
    for act in action_space():
        if mcts.Q[(history,act)] > max_q:
            max_q = mcts.Q[(history,act)]
            opt_act = act
        tmp_qval.append(mcts.Q[(history,act)])
    if tmp_qval[0] == tmp_qval[1]:
        #no optimal action, choose random action
        opt_act = np.random.randint(0,2)
    return opt_act

new_game = generate_new_game(initial_state())
print(rollout(new_game,2,naive_policy))

print(selectaction(MCTS(),[],2,10))

"""
def action_space(): return [0,1]

def state_space(max_card_val,max_hand_val):
    all_deck = list(itertools.product([2, 3, 4], repeat=max_card_val))
    return [[deck,i,j] for i in range(1,max_hand_val+1) for j in range(1,max_hand_val+1) for deck in all_deck]

def observation_space(max_card_val):
    return [[i,j] for i in range(1,max_hand_val+1) for j in range(1,max_hand_val+1) ]

def transition(s,a,s_p):
    init_s = np.array(s[0])
    new_s = np.array(s_p[0])
    if a == 1:
        if (init_s - new_s).dot(np.ones_like(init_s)) != 1:
            return 0
        else:
            for card in range(0,len(init_s)):
                if init_s[card] != new_s[card]:
                    if s_p[1] != s[1] + card+4:
                        return 0
                    if s[2] != s_p[2]:
                        return 0
                    if init_s[card] == 1:
                        return 0
                    else:
                        return init_s[card]/sum(init_s)
    else:
        diff = init_s - new_s
        if s_p[1] != s[1] or s_p[2] < 17:
            return 0
        value_added = 0.
        added_count = 0
        num_count = 1.
        for card in range(0,len(diff)):
            if diff[card] < 0:
                return 0
            value_added += (card+4)*diff[card]
            added_count += diff[card]
            num_count = num_count* ncr(init_s[card],diff[card])
        if s_p[2] != s[2] + value_added:
            return 0
        else:
            return num_count / ncr(sum(init_s),added_count)
fullspace = state_space(8,31)

print(transition([[4,4,4,4,4,4,4,4],0,0],1,[[4,4,4,4,4,4,4,3],11,0]))
print(transition([[4,4,4,4,4,4,4,4],0,0],0,[[4,4,3,4,4,4,4,3],0,17]))

print(transition([[3,4,4,4,4,4,2,4],14,10],1,[[2,4,4,4,4,4,2,4],18,10]))

print(sum([transition([[3,4,4,4,4,4,2,4],14,10],1,s_p) for s_p in fullspace]))
"""
