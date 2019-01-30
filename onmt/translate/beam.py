from __future__ import division
import torch
from onmt.translate import penalties
from scipy.cluster.vq import kmeans, vq, whiten
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import math

## Calculates distance between 2 vectors
def calc_distance(x, y):
    nx = np.asarray(x).reshape(1, -1)
    ny = np.asarray(y).reshape(1, -1)
    dist = euclidean_distances(nx, ny)
    return dist[0][0]


class Beam(object):
    """
    Class for managing the internals of the beam search process.

    Takes care of beams, back pointers, and scores.

    Args:
       size (int): beam size
       pad, bos, eos (int): indices of padding, beginning, and ending.
       n_best (int): nbest size to use
       cuda (bool): use gpu
       global_scorer (:obj:`GlobalScorer`)
    """

    def __init__(self, size, pad, bos, eos, vocab,
                 n_best=1, cuda=False,
                 global_scorer=None,
                 min_length=0,
                 stepwise_penalty=False,
                 block_ngram_repeat=0,
                 exclusion_tokens=set(),
                 prev_hyps=[],
                 hamming_dist=1,
                 num_clusters=1,
                 embeddings=[],
                 k_per_cand=0,
                 hamming_penalty=0.0):

        self.size = size
        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        self.all_scores = []

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        self.next_ys = [self.tt.LongTensor(size)
                        .fill_(pad)]
        self.next_ys[0][0] = bos

        # Has EOS topped the beam yet.
        self._eos = eos
        self.eos_top = False

        # The attentions (matrix) for each time.
        self.attn = []

        # Time and k pair for finished.
        self.finished = []
        self.n_best = n_best

        # Information for global scoring.
        self.global_scorer = global_scorer
        self.global_state = {}

        # Minimum prediction length
        self.min_length = min_length

        # Apply Penalty at every step
        self.stepwise_penalty = stepwise_penalty
        self.block_ngram_repeat = block_ngram_repeat
        self.exclusion_tokens = exclusion_tokens

        ## For debugging purposes
        self.current_beam_str = []
        self.vocab = vocab

        ## Arguments for clustered beam search
        self.num_clusters = num_clusters
        self.embeddings = embeddings

        ## Arguments for iterative beam search
        self.prev_hyps = prev_hyps
        self.hamming_dist = hamming_dist

        # Argument for k_per_cand
        self.k_per_cand = k_per_cand

        # Argument for diverse beam search
        self.hamming_penalty = hamming_penalty


    def get_current_state(self):
        "Get the outputs for the current timestep."
        return self.next_ys[-1]

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]

    def advance(self, word_probs, attn_out, current_beam_str, current_step):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attn_out`: Compute and update the beam search.

        Parameters:

        * `word_probs`- probs of advancing from the last step (K x words)
        * `attn_out`- attention at the last step

        Returns: True if beam search is complete.
        """
        num_words = word_probs.size(1)
        if self.stepwise_penalty:
            self.global_scorer.update_score(self, attn_out)
        # force the output to be longer than self.min_length
        cur_len = len(self.next_ys)
        if cur_len < self.min_length:
            for k in range(len(word_probs)):
                word_probs[k][self._eos] = -1e20
        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            try:
                beam_scores = word_probs + self.scores.unsqueeze(1)
            except RuntimeError:
                beam_scores = word_probs.double().cuda() + self.scores.unsqueeze(1).double().cuda()
            # Don't let EOS have children.
            for i in range(self.next_ys[-1].size(0)):
                if self.next_ys[-1][i] == self._eos:
                    beam_scores[i] = -1e20

            # Block ngram repeats
            if self.block_ngram_repeat > 0:
                ngrams = []
                le = len(self.next_ys)
                for j in range(self.next_ys[-1].size(0)):
                    hyp, _ = self.get_hyp(le - 1, j)
                    ngrams = set()
                    fail = False
                    gram = []
                    for i in range(le - 1):
                        # Last n tokens, n = block_ngram_repeat
                        gram = (gram +
                                [hyp[i].item()])[-self.block_ngram_repeat:]
                        # Skip the blocking if it is in the exclusion list
                        if set(gram) & self.exclusion_tokens:
                            continue
                        if tuple(gram) in ngrams:
                            fail = True
                        ngrams.add(tuple(gram))
                    if fail:
                        beam_scores[j] = -10e20
        else:
            beam_scores = word_probs[0]
        flat_beam_scores = beam_scores.view(-1)

        ## K_PER_CAND CODE
        if current_step > 0 and self.k_per_cand != 0:
            scores, scores_id = flat_beam_scores.sort(0, descending=True)
            prev_k = scores_id / num_words
            next_k = scores_id - prev_k * num_words

            '''
            #### FOR DEBUGGING (DELETE LATER)
            print("\nORIGINAL BEAM: ")
            for i in range(self.size):
                if current_step == 0:
                    toks = ["\t", self.vocab.itos[next_k[i].item()]]
                else:
                    toks = current_beam_str[prev_k[i]].split(" ") + ["\t", self.vocab.itos[next_k[i].item()]]
                ind = next_k[i].item()   
                try:
                    print(" ".join(toks) + "\t" + str(ind) + "\t" + str(scores[i].item()))
                except UnicodeEncodeError:
                    continue
            ####
            '''

            ## Keeps track of number of candidates from the same candidate on previous beam
            prev_beam_counts = dict()
            for b in range(self.size):
                prev_beam_counts[b] = 0

            scores_temp = []
            prev_k_temp = []
            next_k_temp = []
            for i in range(self.size):
                indices = ((prev_k == i).nonzero())[:self.k_per_cand]

                for i in indices:
                    scores_temp.append(scores[i].item())
                    prev_k_temp.append(prev_k[i])
                    next_k_temp.append(next_k[i])

            scores_temp = torch.from_numpy(np.array(scores_temp, dtype='double')).cuda()
            best_scores, scores_id = scores_temp.topk(self.size, 0, True, True)
            prev_k = torch.from_numpy(np.array([prev_k_temp[i].item() for i in scores_id], dtype='int32')).type(torch.LongTensor).cuda()
            next_k = torch.from_numpy(np.array([next_k_temp[i].item() for i in scores_id], dtype='int32')).type(torch.LongTensor).cuda()

            '''
            ####### FOR DEBUGGING (DELETE LATER)
            print("\nBEAM AFTER ONLY CONSIDERING TOP K FOR EACH CANDIDATE: ")
            for i in range(self.size):
                if current_step == 0:
                    toks = ["\t", self.vocab.itos[next_k[i].item()]]
                else:
                    toks = current_beam_str[prev_k[i]].split(" ") + ["\t", self.vocab.itos[next_k[i].item()]]
                ind = next_k[i].item()   
                try:
                    print(" ".join(toks) + "\t" + str(ind) + "\t" + str(best_scores[i].item()))
                except UnicodeEncodeError:
                    continue
            ####
            '''

            self.all_scores.append(self.scores)
            self.scores = best_scores
            self.prev_ks.append(prev_k)
            self.next_ys.append(next_k)
            self.attn.append(attn_out.index_select(0, prev_k))
            self.global_scorer.update_global_state(self)

            for i in range(self.next_ys[-1].size(0)):
                if self.next_ys[-1][i] == self._eos:
                    global_scores = self.global_scorer.score(self, self.scores)
                    s = global_scores[i]
                    self.finished.append((s, len(self.next_ys) - 1, i))

            # End condition is when top-of-beam is EOS and no global score.
            if self.next_ys[-1][0] == self._eos:
                self.all_scores.append(self.scores)
                self.eos_top = True

        ## DIVERSE BEAM SEARCH CODE
        elif current_step > 0 and self.hamming_penalty > 0.0:
            scores, scores_id = flat_beam_scores.sort(0, descending=True)
            prev_k = scores_id / num_words
            next_k = scores_id - prev_k * num_words

            '''
            #### FOR DEBUGGING (DELETE LATER)
            print("\nORIGINAL BEAM: ")
            for i in range(self.size):
                if current_step == 0:
                    toks = ["\t", self.vocab.itos[next_k[i].item()]]
                else:
                    toks = current_beam_str[prev_k[i]].split(" ") + ["\t", self.vocab.itos[next_k[i].item()]]
                ind = next_k[i].item()   
                try:
                    print(" ".join(toks) + "\t" + str(ind) + "\t" + str(scores[i].item()))
                except UnicodeEncodeError:
                    continue
            ####
            '''

            indices = []
            for i in range(self.size):
                indices += ((prev_k == i).nonzero())[:self.size]
                print(indices)

            ## Saves counts of words seen in candidates so far
            word_counts = dict()
            '''
            prev_beam_counts = dict()
            for b in range(self.size):
                prev_beam_counts[b] = 0
            '''

            scores_temp = []
            prev_k_temp = []
            next_k_temp = []
            scores_orig = []

            for i in indices:
                #if prev_beam_counts[prev_k[i].item()] < self.size:
                ## Gets word count of generated token
                tok = self.vocab.itos[next_k[i].item()]
                try:
                    c = word_counts[tok]
                    word_counts[tok] += 1
                except KeyError:
                    c = 0
                    word_counts[tok] = 1

                scores_temp.append(scores[i] - self.hamming_penalty*c)
                scores_orig.append(scores[i])

                prev_k_temp.append(prev_k[i])
                next_k_temp.append(next_k[i])
                

                '''
                prev_beam_counts[prev_k[i].item()] += 1
                else:
                    key_min = min(prev_beam_counts.keys(), key=(lambda k: prev_beam_counts[k]))
                    if prev_beam_counts[key_min] == self.size:
                        print(i)
                        break
                '''

            print(len(scores_temp))
            best_scores = torch.from_numpy(np.array(scores_temp, dtype='double')).cuda()
            best_scores, scores_id = best_scores.topk(self.size, 0, True, True)

            prev_k = torch.from_numpy(np.array([prev_k_temp[i].item() for i in scores_id], dtype='int32')).type(torch.LongTensor).cuda()
            next_k = torch.from_numpy(np.array([next_k_temp[i].item() for i in scores_id], dtype='int32')).type(torch.LongTensor).cuda()

            '''
            ####### FOR DEBUGGING (DELETE LATER)
            print("\nBEAM AFTER DIVERSE BEAM SEARCH")
            for i in range(self.size):
                if current_step == 0:
                    toks = ["\t", self.vocab.itos[next_k[i].item()]]
                else:
                    toks = current_beam_str[prev_k[i]].split(" ") + ["\t", self.vocab.itos[next_k[i].item()]]
                ind = next_k[i].item()   
                try:
                    print(" ".join(toks) + "\t" + str(ind) + "\t" + str(best_scores[i].item()))
                except UnicodeEncodeError:
                    continue
            ####
            '''

            self.all_scores.append(self.scores)
            self.scores = best_scores
            self.prev_ks.append(prev_k)
            self.next_ys.append(next_k)
            self.attn.append(attn_out.index_select(0, prev_k))
            self.global_scorer.update_global_state(self)

            for i in range(self.next_ys[-1].size(0)):
                if self.next_ys[-1][i] == self._eos:
                    global_scores = self.global_scorer.score(self, self.scores)
                    s = global_scores[i]
                    self.finished.append((s, len(self.next_ys) - 1, i))

            # End condition is when top-of-beam is EOS and no global score.
            if self.next_ys[-1][0] == self._eos:
                self.all_scores.append(self.scores)
                self.eos_top = True

        ## CLUSTERED BEAM SEARCH CODE
        elif self.num_clusters > 1:
            scores, scores_id = flat_beam_scores.topk(self.size*2, 0, True, True)
            prev_k = scores_id / num_words
            next_k = scores_id - prev_k * num_words

            '''
            #### FOR DEBUGGING (DELETE LATER)
            print("\nORIGINAL BEAM: ")
            for i in range(self.size):
                if current_step == 0:
                    toks = ["\t", self.vocab.itos[next_k[i].item()]]
                else:
                    toks = current_beam_str[prev_k[i]].split(" ") + ["\t", self.vocab.itos[next_k[i].item()]]
                ind = next_k[i].item()   
                try:
                    print(" ".join(toks) + "\t" + str(ind) + "\t" + str(scores[i].item()))
                except UnicodeEncodeError:
                    continue
            ####
            '''

            ## Get vector representations of candidates
            embeds = []
            for i in range(len(scores)):
                if current_step == 0:
                    toks = [self.vocab.itos[next_k[i].item()]]
                else:
                    toks = current_beam_str[prev_k[i]].split(" ") + [self.vocab.itos[next_k[i].item()]]

                cand_embeds = [self.embeddings[t] for t in toks]
                embeds.append(sum(cand_embeds)/len(cand_embeds))
            std_embeds = whiten(embeds)

            ## Run K-means to cluster candidates into K clusters
            centroids,_ = kmeans(std_embeds, self.num_clusters)
            cluster_labels = []
            for e in std_embeds:
                min_distance = 1e10
                label = -1
                for i, c in enumerate(centroids):
                    dist = calc_distance(c, e)

                    if dist < min_distance:
                        min_distance = dist
                        label = i
                cluster_labels.append(label)

            '''
            #### FOR DEBUGGING (DELETE LATER)
            print("\nBEAM CLUSTERS: ")
            for c in range(max(cluster_labels)+1):
                print("Cluster: " + str(c))
                for i in range(len(prev_k)):
                    if cluster_labels[i] == c:
                        if current_step == 0:
                            toks = ["\t", self.vocab.itos[next_k[i].item()]]
                        else:
                            toks = current_beam_str[prev_k[i]].split(" ") + ["\t", self.vocab.itos[next_k[i].item()]]
                        ind = next_k[i].item()
                        try:
                           print(" ".join(toks) + "\t" + str(ind) + "\t" + str(scores[i].item()))
                        except UnicodeEncodeError:
                            continue
            #######
            '''

            ## Get top BEAM_SIZE/K candidates from each cluster
            cscores, cprev_k, cnext_k = [], [], []
            cluster_counts = [0 for i in range(self.num_clusters)]
            indices = []

            for i, l in enumerate(cluster_labels):
                if cluster_counts[l] < math.ceil(self.size / self.num_clusters):
                    cscores.append(scores[i])
                    cprev_k.append(prev_k[i])
                    cnext_k.append(next_k[i])
                    indices.append(i)
                    cluster_counts[l] += 1
                elif min(cluster_counts) == math.ceil(self.size / self.num_clusters):
                    break
                else:
                    continue

            ## If there aren't enough in each cluster, add candidates with highest scores
            if len(cscores) < self.size:
                while len(cscores) != self.size:
                    for i, l in enumerate(cluster_labels):
                        if i not in indices:
                            cscores.append(scores[i])
                            cprev_k.append(prev_k[i])
                            cnext_k.append(next_k[i])
                            indices.append(i)
                            break

            best_scores = torch.from_numpy(np.array(cscores, dtype='double')).cuda()
            best_scores, scores_id = best_scores.sort(0, descending=True)
            prev_k = torch.from_numpy(np.array([cprev_k[i] for i in scores_id], dtype='int32')).type(torch.LongTensor).cuda()
            next_k = torch.from_numpy(np.array([cnext_k[i] for i in scores_id], dtype='int32')).type(torch.LongTensor).cuda()

            '''
            #### FOR DEBUGGING (DELETE LATER)
            print("\nBEAM AFTER CLUSTERED BEAM SEARCH: ")
            for i in range(len(prev_k)):
                if current_step == 0:
                    toks = ["\t", self.vocab.itos[next_k[i].item()]]
                else:
                    toks = current_beam_str[prev_k[i]].split(" ") + ["\t", self.vocab.itos[next_k[i].item()]]
                ind = next_k[i].item()
                try:
                   print(" ".join(toks) + "\t" + str(ind) + "\t" + str(scores[i].item()))
                except UnicodeEncodeError:
                    continue
            #######
            '''

            self.all_scores.append(self.scores)
            self.scores = best_scores
            self.prev_ks.append(prev_k)
            self.next_ys.append(next_k)
            self.attn.append(attn_out.index_select(0, prev_k))
            self.global_scorer.update_global_state(self)

            for i in range(self.next_ys[-1].size(0)):
                if self.next_ys[-1][i] == self._eos:
                    global_scores = self.global_scorer.score(self, self.scores)
                    s = global_scores[i]
                    self.finished.append((s, len(self.next_ys) - 1, i))

            # End condition is when top-of-beam is EOS and no global score.
            if self.next_ys[-1][0] == self._eos:
                self.all_scores.append(self.scores)
                self.eos_top = True

        ## ITERATIVE BEAM SEARCH CODE
        elif self.prev_hyps != []:
            scores, scores_id = flat_beam_scores.sort(0, descending=True)
            prev_k = scores_id / num_words
            next_k = scores_id - prev_k * num_words

            '''
            #### FOR DEBUGGING (DELETE LATER)
            print("\nORIGINAL BEAM: ")
            for i in range(self.size):
                if current_step == 0:
                    toks = ["\t", self.vocab.itos[next_k[i].item()]]
                else:
                    toks = current_beam_str[prev_k[i]].split(" ") + ["\t", self.vocab.itos[next_k[i].item()]]
                ind = next_k[i].item()   
                try:
                    print(" ".join(toks) + "\t" + str(ind) + "\t" + str(scores[i].item()))
                except UnicodeEncodeError:
                    continue
            ####
            '''

            ## Removes all candidates already found in a previous beam search
            scores_temp = []
            prev_k_temp = []
            next_k_temp = []

            non_dups = 0
            for i in range(len(scores)):
                if current_step == 0:
                    toks = " ".join([self.vocab.itos[next_k[i].item()]])
                else:
                    toks = " ".join(current_beam_str[prev_k[i]].split(" ") + [self.vocab.itos[next_k[i].item()]])

                if toks not in self.prev_hyps and toks != '<unk>':
                    scores_temp.append(scores[i].item())
                    non_dups += 1
                    prev_k_temp.append(prev_k[i].item())
                    next_k_temp.append(next_k[i].item())

                    if non_dups >= self.size:
                        #scores_temp += [s.item() for s in scores[i+1:]]
                        break

            best_scores = torch.from_numpy(np.array(scores_temp, dtype='double')).cuda()

            #best_scores, best_scores_id = scores.topk(self.size, 0, True, True)

            prev_k = torch.from_numpy(np.array(prev_k_temp, dtype='int32')).type(torch.LongTensor).cuda()
            next_k = torch.from_numpy(np.array(next_k_temp, dtype='int32')).type(torch.LongTensor).cuda()

            '''
            #### FOR DEBUGGING (DELETE LATER)
            print("\nBEAM AFTER ITERATIVE BEAM SEARCH: ")
            for i in range(len(prev_k)):
                if current_step == 0:
                    toks = ["\t", self.vocab.itos[next_k[i].item()]]
                else:
                    toks = current_beam_str[prev_k[i]].split(" ") + ["\t", self.vocab.itos[next_k[i].item()]]
                ind = next_k[i].item()
                try:
                   print(" ".join(toks) + "\t" + str(ind) + "\t" + str(best_scores[i].item()))
                except UnicodeEncodeError:
                    continue
            #######
            '''

            self.all_scores.append(self.scores)
            self.scores = best_scores

            self.prev_ks.append(prev_k)
            self.next_ys.append(next_k)
            self.attn.append(attn_out.index_select(0, prev_k))
            self.global_scorer.update_global_state(self)

            for i in range(self.next_ys[-1].size(0)):
                if self.next_ys[-1][i] == self._eos:
                    global_scores = self.global_scorer.score(self, self.scores)
                    s = global_scores[i]
                    self.finished.append((s, len(self.next_ys) - 1, i))

            # End condition is when top-of-beam is EOS and no global score.
            if self.next_ys[-1][0] == self._eos:
                self.all_scores.append(self.scores)
                self.eos_top = True

        ## REGULAR BEAM SEARCH CODE
        else:
            best_scores, best_scores_id = flat_beam_scores.topk(self.size, 0,
                                                                True, True)

            self.all_scores.append(self.scores)
            self.scores = best_scores

            # best_scores_id is flattened beam x word array, so calculate which
            # word and beam each score came from
            prev_k = best_scores_id / num_words
            self.prev_ks.append(prev_k)
            self.next_ys.append((best_scores_id - prev_k * num_words))
            self.attn.append(attn_out.index_select(0, prev_k))
            self.global_scorer.update_global_state(self)

            for i in range(self.next_ys[-1].size(0)):
                if self.next_ys[-1][i] == self._eos:
                    global_scores = self.global_scorer.score(self, self.scores)
                    s = global_scores[i]
                    self.finished.append((s, len(self.next_ys) - 1, i))

            # End condition is when top-of-beam is EOS and no global score.
            if self.next_ys[-1][0] == self._eos:
                self.all_scores.append(self.scores)
                self.eos_top = True




    def done(self):
        return self.eos_top and len(self.finished) >= self.n_best

    def sort_finished(self, minimum=None):
        if minimum is not None:
            i = 0
            # Add from beam until we have minimum outputs.
            while len(self.finished) < minimum:
                global_scores = self.global_scorer.score(self, self.scores)
                s = global_scores[i]
                self.finished.append((s, len(self.next_ys) - 1, i))
                i += 1

        self.finished.sort(key=lambda a: -a[0])
        scores = [sc for sc, _, _ in self.finished]
        ks = [(t, k) for _, t, k in self.finished]
        return scores, ks

    ## ADDED CODE: Gets current beam (without changing output)
    def get_current_beam_str(self, beam_size):
        prev_beam = list(self.current_beam_str)
        self.current_beam_str = []

        ## Keeps all candidates from previous beam if they are finished
        for (s, step, rank, fin) in prev_beam:
            if fin:
                self.current_beam_str.append((s, step, rank, fin))

        ## Adds all candidates from current beam, even if they are not finished
        for i in range(beam_size):
            global_scores = self.global_scorer.score(self, self.scores)
            s = global_scores[i]

            if self.next_ys[len(self.next_ys)-1][i] == self._eos:
                fin = True
            else:
                fin = False

            self.current_beam_str.append((s, len(self.next_ys) - 1, i, fin))

        ## Sorts the beam
        self.current_beam_str.sort(key=lambda a: -a[0])
        scores = [sc for sc, _, _, _ in self.current_beam_str]
        ks = [(t, k) for _, t, k, _ in self.current_beam_str]
        fins = [f for _, _, _, f in self.current_beam_str]
        return scores, ks, fins

    def get_hyp(self, timestep, k):
        """
        Walk back to construct the full hypothesis.
        """
        hyp, attn = [], []
        for j in range(len(self.prev_ks[:timestep]) - 1, -1, -1):
            hyp.append(self.next_ys[j + 1][k])
            attn.append(self.attn[j][k])
            k = self.prev_ks[j][k]
        return hyp[::-1], torch.stack(attn[::-1])


class GNMTGlobalScorer(object):
    """
    NMT re-ranking score from
    "Google's Neural Machine Translation System" :cite:`wu2016google`

    Args:
       alpha (float): length parameter
       beta (float):  coverage parameter
    """

    def __init__(self, opt):
        self.alpha = opt.alpha
        self.beta = opt.beta
        penalty_builder = penalties.PenaltyBuilder(opt.coverage_penalty,
                                                   opt.length_penalty)
        # Term will be subtracted from probability
        self.cov_penalty = penalty_builder.coverage_penalty()
        # Probability will be divided by this
        self.length_penalty = penalty_builder.length_penalty()

    def score(self, beam, logprobs):
        """
        Rescores a prediction based on penalty functions
        """
        normalized_probs = self.length_penalty(beam,
                                               logprobs,
                                               self.alpha)
        if not beam.stepwise_penalty:
            penalty = self.cov_penalty(beam,
                                       beam.global_state["coverage"],
                                       self.beta)
            normalized_probs -= penalty

        return normalized_probs

    def update_score(self, beam, attn):
        """
        Function to update scores of a Beam that is not finished
        """
        if "prev_penalty" in beam.global_state.keys():
            beam.scores.add_(beam.global_state["prev_penalty"])
            penalty = self.cov_penalty(beam,
                                       beam.global_state["coverage"] + attn,
                                       self.beta)
            beam.scores.sub_(penalty)

    def update_global_state(self, beam):
        "Keeps the coverage vector as sum of attentions"
        if len(beam.prev_ks) == 1:
            beam.global_state["prev_penalty"] = beam.scores.clone().fill_(0.0)
            beam.global_state["coverage"] = beam.attn[-1]
            self.cov_total = beam.attn[-1].sum(1)
        else:
            self.cov_total += torch.min(beam.attn[-1],
                                        beam.global_state['coverage']).sum(1)
            beam.global_state["coverage"] = beam.global_state["coverage"] \
                .index_select(0, beam.prev_ks[-1]).add(beam.attn[-1])

            prev_penalty = self.cov_penalty(beam,
                                            beam.global_state["coverage"],
                                            self.beta)
            beam.global_state["prev_penalty"] = prev_penalty
