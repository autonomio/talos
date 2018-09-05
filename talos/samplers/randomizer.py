import numpy as np

from .sobol.sobol_seq import i4_sobol_generate
from .hypercube.hycusampling import halton, korobov_design_matrix
from .hypercube.hycusampling import improved_lhd_matrix, lhd_matrix
from .lhs_sudoku import sudoku
from .quantum import cached_generator, randint
from .ambient import RandomOrgClient


class Randomizer:

    def __init__(self, length, n):

        self.len = length
        self.n = n

    def uniform_mersenne(self):

        '''Regular uniform / pseudorandom sequence'''

        out = list(range(self.len))
        np.random.shuffle(out)
        return out[:self.n]

    def uniform_crypto(self):

        from secrets import randbelow

        '''Cryptographically sound pseudorandom sequence'''

        out = []
        i = 0
        while i < self.n:
            num = randbelow(self.len)
            if num not in out:
                i += 1
                out.append(num)

        return out

    def latin_sudoku(self, dims=2, sudoku_boxes=1):

        '''Latin Hypercube with Sudoku-style Constraint.
        M. D. McKay, R. J. Beckman, W. J. Conover, 1979.

        dims :: number of dimensions
        sudoku_boxes :: number of boxes to use as constraint

        '''

        n = int(self.len / sudoku_boxes)

        if self.len % sudoku_boxes != 0:
            raise ValueError('Index len must be divisible by sudoku_boxes')

        out = sudoku.sample(dims, sudoku_boxes, n)
        out = [i[0] for i in out[0]]

        return out[:self.n]

    def latin_improved(self):

        out = [i[0] for i in improved_lhd_matrix(self.len, 1)]
        return out[:self.n]

    def latin_matrix(self):

        out = [i[0] for i in lhd_matrix(self.len, 1)]
        return out[:self.n]

    def sobol(self):

        '''Creates an index based on Sobol Sequence'''

        org = [i[0] for i in i4_sobol_generate(1, self.len)]
        return self._match_index(org)[:self.n]

    def halton(self):

        '''Creates an index based on Halton Sequence'''

        org = [i[0] for i in halton(self.len, 1, 5)]
        return self._match_index(org)[:self.n]

    def korobov_matrix(self):

        '''Returns a 1-d array of integeres in the Korobov Design Matrix'''

        out = [i for i in korobov_design_matrix(self.len, 2)[:, 1]]
        return out[:self.n]

    def ambience(self):

        '''An ambient sound based TRNG using RANDOM.ORG API'''

        out = RandomOrgClient('c7d1e0b0-e57c-4fb8-9d5b-382ec9de5a89')
        return out.generate_integers(self.n, 0, self.len)

    def quantum(self):

        '''Quantum Random Number Generator

        NOTE: this method can only return 1024 random numbers.

        DESCRIPTION
        ===========

        The random numbers are generated in real-time in ANU lab by measuring
        the quantum fluctuations of the vacuum. The vacuum is described very
        differently in the quantum mechanical context than in the classical
        context. Traditionally, a vacuum is considered as a space that is
        empty of matter or photons. Quantum mechanically, however, that same
        space resembles a sea of virtual particles appearing and disappearing
        all the time. This result is due to the fact that the vacuum still
        possesses a zero-point energy. Consequently, the electromagnetic
        field of the vacuum exhibits random fluctuations in phase and amplitude
        at all frequencies. By carefully measuring these fluctuations, we are
        able to generate ultra-high bandwidth random numbers.

        EXAMPLE
        =======

        test = quantum_random(200, minimum=10, maximum=20)
        randhist(test)

        PARAMETERS
        ==========

        n = number of integer values to return

        minimum = min value for integers

        maximum = max value for integers

        '''

        out = []

        gen = cached_generator()

        for i in range(self.n):
            out.append(int(randint(min=0, max=self.len, generator=gen)))

        return out

    def _match_index(self, org):

        '''Helper to match sequence with index and
        reorganize index accordingly.'''

        temp = np.array(list(zip(org, list(range(self.len)))))
        out = temp[temp[:, 0].argsort()][:, 1].astype(int).tolist()

        return out
