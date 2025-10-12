import numpy as np
from typing import List, Tuple, Optional

class VQ:
    def __init__(self, cfg_vq: tuple[int], d_out: int, vq_type: str = None, VLEN: int = 4096, BW: int = 4, seed: int = 42):
        """
        VQ<d,m,n,k>
        D: dimension of the input vector
        m: number of subvectors
        n: number of codebooks
        k: number of clusters (LUT/codebook size)
        """
        self.d_dim = cfg_vq[0]
        self.n_subvec = cfg_vq[1]
        self.n_codebook = cfg_vq[2]
        self.n_cluster = cfg_vq[3]
        self.d_subvec = self.d_dim // self.n_subvec
        self.d_out = d_out
        self.vq_type = vq_type
        self.VLEN = VLEN
        self.BW = np.log2(self.n_cluster)
        self.dtype = np.float16

        np.random.seed(seed)
        # TMAC-style VQ initializes codebook to -16 to 15 values
        self.init_codebook()
        self.init_codeword()
        self.dequantize()

        self.perf_cnt = {
            "offchip_access_lut": 0,
            "offchip_access_psum": 0,
            "offchip_access_codeword": 0,
            "offchip_access_codebook": 0,
            "offchip_access_inp": 0,
        }

    def init_codebook(self):
        if self.vq_type == 'TMAC':
            # TMAC-style bit-serial GEMM initializes codebook to [-1, -1, -1, -1] to [1, 1, 1, 1]
            codebook_base = np.zeros((self.n_cluster, self.d_subvec))
            for i in range(self.n_cluster):
                # Convert i to a d_subvec-dim vector based on its binary values
                bin_vec = np.array([(i >> bit) & 1 for bit in range(self.d_subvec)]) * 2 - 1  # [-1, 1] for each bit
                codebook_base[i, :] = bin_vec
            self.codebook = codebook_base.reshape(1, 1, self.n_cluster, self.d_subvec) * np.ones((self.n_subvec, self.n_codebook, 1, 1))

            # Scaling factor for bit serial
            bs_scaling = (2 ** np.arange(self.n_codebook)).reshape(1, -1, 1, 1)
            self.codebook = self.codebook * bs_scaling
        else:
            self.codebook = np.random.randn(self.n_subvec, self.n_codebook, self.n_cluster, self.d_subvec)

        self.codebook = self.codebook.astype(self.dtype)

    def init_codeword(self):
        if self.vq_type == 'TMAC':
            # Dequantize weight when running TMAC-style VQ
            self.codeword = np.random.randint(2, size=(self.d_out, self.n_subvec, self.d_subvec, self.n_codebook), dtype=np.int8)
            self.weight_mat = (2*self.codeword-1) * (2 ** np.arange(self.n_codebook)).reshape(1, 1, 1, -1)
            self.weight_mat = self.weight_mat.sum(axis=-1).reshape(self.d_out, self.d_dim).astype(np.int8)
            self.codeword = (self.codeword * (2**np.arange(self.d_subvec)).reshape(1, 1, self.d_subvec, 1)).sum(axis=-2)
        else:
            self.codeword = np.random.randint(0, self.n_cluster, (self.d_out, self.n_subvec, self.n_codebook), dtype=np.uint8)
            
        self.codeword = self.codeword.astype(np.uint8)
    
    def dataflow_sim(self, x: np.ndarray, dataflow: str = ''):
        def get_tiling_parms():
            self.n_codebook_tile = self.n_codebook
            while (self.VLEN // 16 // self.n_codebook_tile // self.n_cluster == 0):
                self.n_codebook_tile = self.n_codebook_tile // 2

            # ADT tiling parameters
            self.n_cluster_tile = self.VLEN // 16 // self.d_subvec // self.n_codebook_tile 

            # LUT tiling parameters
            self.LMUL = 16 // self.BW
            self.n_subvec_tile = int(self.LMUL * self.VLEN) // 16 // self.n_codebook_tile // self.n_cluster
            self.n_cw_tile = int(self.n_cluster * 16 // self.BW)
            print(f"ADT tiling: n_codebook_tile: {self.n_codebook_tile}, n_cluster_tile: {self.n_cluster_tile}")
            print(f"LUT tiling: LMUL: {self.LMUL}, n_subvec_tile: {self.n_subvec_tile}, n_cw_tile: {self.n_cw_tile}")

        def _load_codeword(inp: np.ndarray):
            self.perf_cnt["offchip_access_codeword"] += (inp.size * self.BW) 
            return inp

        def _load_lut(inp: np.ndarray):
            self.perf_cnt["offchip_access_lut"] += (inp.size * 16) 
            return inp

        def _load_psum(inp: np.ndarray):
            self.perf_cnt["offchip_access_psum"] += (inp.size * 16) 
            return inp

        def _compute_lut(inp: np.ndarray):
            self.compute_lut(inp)
            self.perf_cnt["offchip_access_inp"] += (inp.size * 16) 
            self.perf_cnt["offchip_access_codebook"] += (self.codebook.size * 16) 
            return self.lut

        d_in, _ = x.shape
        out_mat = np.zeros((d_in, self.d_out))

        get_tiling_parms()

        d_in, _ = x.shape
        out_mat = np.zeros((d_in, self.d_out))

        if dataflow == 'OMND':
            # Dataflow used in LUT-DLA
            lut_inp = np.zeros((d_in, self.n_subvec, self.n_codebook, self.n_cluster))
            for i in range(d_in):
                lut_inp[i, :, :, :] = _load_lut(_compute_lut(x[i, :]))

            for j in range(0, self.d_out, self.n_cw_tile):
                for m in range(0, self.n_subvec, self.n_subvec_tile):
                    for n in range(0, self.n_codebook, self.n_codebook_tile):
                        cw_ = _load_codeword(self.codeword[j:j+self.n_cw_tile, m:m+self.n_subvec_tile, n:n+self.n_codebook_tile])

                        for i in range(d_in):
                            _load_psum(out_mat[i,j:j+self.n_cw_tile])
                            lut_ = _load_lut(lut_inp[i, m:m+self.n_subvec_tile, n:n+self.n_codebook_tile, :])

                            # out_mat[i,j:j] += lut_[cw_]
        elif dataflow == 'DMNO':
            lut_inp = np.zeros((d_in, self.n_subvec, self.n_codebook, self.n_cluster))
            for i in range(d_in):
                lut_inp[i, :, :, :] = _load_lut(_compute_lut(x[i, :]))

            for i in range(d_in):
                for m in range(0, self.n_subvec, self.n_subvec_tile):
                    for n in range(0, self.n_codebook, self.n_codebook_tile):
                        lut_ = _load_lut(lut_inp[i, m:m+self.n_subvec_tile, n:n+self.n_codebook_tile, :])

                        for j in range(0, self.d_out, self.n_cw_tile):
                            _load_psum(out_mat[i,j:j+self.n_cw_tile])

                            cw_ = _load_codeword(self.codeword[j:j+self.n_cw_tile, m:m+self.n_subvec_tile, n:n+self.n_codebook_tile])
                            # out_mat[i,j] += lut_[cw_]
        elif dataflow == 'MNOD':
            lut_inp = np.zeros((d_in, self.n_subvec, self.n_codebook, self.n_cluster))
            for i in range(d_in):
                lut_inp[i, :, :, :] = _load_lut(_compute_lut(x[i, :]))

            for m in range(0, self.n_subvec, self.n_subvec_tile):
                for n in range(0, self.n_codebook, self.n_codebook_tile):
                    for j in range(0, self.d_out, self.n_cw_tile):
                        cw_ = _load_codeword(self.codeword[j:j+self.n_cw_tile, m:m+self.n_subvec_tile, n:n+self.n_codebook_tile])
                        for i in range(d_in):
                            _load_psum(out_mat[i,j:j+self.n_cw_tile])
                            lut_ = _load_lut(lut_inp[i, m:m+self.n_subvec_tile, n:n+self.n_codebook_tile, :])
                            # out_mat[i,j] += lut_[cw_]
        elif dataflow == 'MNDO':
            lut_inp = np.zeros((d_in, self.n_subvec, self.n_codebook, self.n_cluster))
            for i in range(d_in):
                # lut_inp[i, :, :, :] = _load_lut(_compute_lut(x[i, :]))
                lut_inp[i, :, :, :] = _compute_lut(x[i, :])

            for m in range(0, self.n_subvec, self.n_subvec_tile):
                for n in range(0, self.n_codebook, self.n_codebook_tile):
                    for i in range(d_in):
                        lut_ = _load_lut(lut_inp[i, m:m+self.n_subvec_tile, n:n+self.n_codebook_tile, :])

                        for j in range(0, self.d_out, self.n_cw_tile):
                            _load_psum(out_mat[i,j:j+self.n_cw_tile])

                            cw_ = _load_codeword(self.codeword[j:j+self.n_cw_tile, m:m+self.n_subvec_tile, n:n+self.n_codebook_tile])
                            # out_mat[i,j] += lut_[cw_]
        elif dataflow == 'ODMN':
            lut_inp = np.zeros((d_in, self.n_subvec, self.n_codebook, self.n_cluster))
            for i in range(d_in):
                lut_inp[i, :, :, :] = _load_lut(_compute_lut(x[i, :]))

            for j in range(0, self.d_out, self.n_cw_tile):
                for i in range(d_in):
                    _load_psum(out_mat[i,j:j+self.n_cw_tile])

                    for m in range(0, self.n_subvec, self.n_subvec_tile):
                        for n in range(0, self.n_codebook, self.n_codebook_tile):
                            lut_ = _load_lut(lut_inp[i, m:m+self.n_subvec_tile, n:n+self.n_codebook_tile, :])
                            cw_ = _load_codeword(self.codeword[j:j+self.n_cw_tile, m:m+self.n_subvec_tile, n:n+self.n_codebook_tile])
                            # out_mat[i,j] += lut_[cw_]
        elif dataflow == 'DOMN':
            lut_inp = np.zeros((d_in, self.n_subvec, self.n_codebook, self.n_cluster))
            for i in range(d_in):
                lut_inp[i, :, :, :] = _load_lut(_compute_lut(x[i, :]))

            for i in range(d_in):
                for j in range(0, self.d_out, self.n_cw_tile):
                    _load_psum(out_mat[i,j:j+self.n_cw_tile])

                    for m in range(0, self.n_subvec, self.n_subvec_tile):
                        for n in range(0, self.n_codebook, self.n_codebook_tile):
                            lut_ = _load_lut(lut_inp[i, m:m+self.n_subvec_tile, n:n+self.n_codebook_tile, :])
                            cw_ = _load_codeword(self.codeword[j:j+self.n_cw_tile, m:m+self.n_subvec_tile, n:n+self.n_codebook_tile])
                            # out_mat[i,j] += lut_[cw_]
        elif dataflow == 'OMND-compact':
            pass
        #     # Dataflow used in LUT-DLA
        #     for j in range(self.d_out):
        #         for m in range(self.n_subvec):
        #             for n in range(self.n_codebook):
        #                 cw_ = _load_codeword(self.codeword[j, m, n])
        #                 for i in range(d_in):
        #                     lut_ = _load_lut(_compute_lut(x[i, :]))
        #                     out_mat[i,j] += lut_[m,n,cw_]
        elif dataflow == 'DMNO-compact':
            pass
            # for i in range(d_in):
            #     lut_ = _load_lut(_compute_lut(x[i, :]))
            #     for m in range(self.n_subvec):
            #         for n in range(self.n_codebook):
            #             for j in range(self.d_out):
            #                 cw_ = _load_codeword(self.codeword[j, m, n])
            #                 out_mat[i,j] += lut_[m,n,cw_]
        elif dataflow == 'MNOD-compact':
            pass
        #     for m in range(self.n_subvec):
        #         for n in range(self.n_codebook):
        #             for j in range(self.d_out):
        #                 cw_ = _load_codeword(self.codeword[j, m, n])
        #                 for i in range(d_in):
        #                     lut_ = _load_lut(_compute_lut(x[i, :]))
        #                     out_mat[i,j] += lut_[m,n,cw_]
        elif dataflow == 'MNDO-compact':
            pass
            # for m in range(self.n_subvec):
            #     for n in range(self.n_codebook):
            #         for i in range(d_in):
            #             lut_ = _load_lut(_compute_lut(x[i, :]))
            #             for j in range(self.d_out):
            #                 cw_ = _load_codeword(self.codeword[j, m, n])
            #                 out_mat[i,j] += lut_[m,n,cw_]
        elif dataflow == 'ODMN-compact':
            pass
            # for j in range(self.d_out):
            #     for i in range(d_in):
            #         lut_ = _load_lut(_compute_lut(x[i, :]))
            #         for m in range(self.n_subvec):
            #             for n in range(self.n_codebook):
            #                 cw_ = _load_codeword(self.codeword[j, m, n])
            #                 out_mat[i,j] += lut_[m,n,cw_]
        elif dataflow == 'DOMN-compact':
            pass
            # for i in range(d_in):
            #     lut_ = _load_lut(_compute_lut(x[i, :]))
            #     for j in range(self.d_out):
            #         for m in range(self.n_subvec):
            #             for n in range(self.n_codebook):
            #                 cw_ = _load_codeword(self.codeword[j, m, n])
            #                 out_mat[i,j] += lut_[m,n,cw_]
        elif dataflow == 'VeLU':
            for i in range(d_in):
                lut_tile = _compute_lut(x[i, :])
                for m in range(0, self.n_subvec, self.n_subvec_tile):

                    for n in range(0, self.n_codebook, self.n_codebook_tile):
                        for j in range(0, self.d_out, self.n_cw_tile):
                            _load_psum(out_mat[i,j:j+self.n_cw_tile])

                            cw_ = _load_codeword(self.codeword[j:j+self.n_cw_tile, m:m+self.n_subvec_tile, n:n+self.n_codebook_tile])
                            # out_mat[i,j] += lut_[m, n, cw_]
        else:
            raise NotImplementedError(f"Dataflow {dataflow} not implemented")

        return out_mat

    def dequantize(self):
        if self.vq_type == 'TMAC':
            pass
        else:
            self.weight_mat = np.zeros((self.d_out, self.n_subvec, self.d_subvec))
            for i in range(self.d_out):
                for m in range(self.n_subvec):
                    for n in range(self.n_codebook): # n_codebook
                        self.weight_mat[i, m] += self.codebook[m, n, self.codeword[i, m, n], :] # n_subvec_dim
            self.weight_mat = self.weight_mat.reshape(self.d_out, self.d_dim).astype(self.dtype)

    def compute_lut(self, x: np.ndarray):
        # x: (1, D)
        # codebook: (M, N, K, d)
        x_reshape = x.reshape(self.n_subvec, 1, 1, self.d_subvec).astype(self.dtype)
        lut = self.codebook * x_reshape # (M, N, K, d) * (M, 1, 1, d) = (M, N, K, d)
        self.lut = lut.sum(axis=-1) # (M, N, K)

    def compute_lut_gemm(self, x: np.ndarray, out_scale: np.ndarray = None):
        # x: (d_in, D)
        # out_scale: (d_out,)
        d_in, _ = x.shape
        out_mat = np.zeros((d_in, self.d_out))

        for i in range(d_in):
            self.compute_lut(x[i, :])
            for j in range(self.d_out):
                for m in range(self.n_subvec):
                    for n in range(self.n_codebook):
                        out_mat[i,j] += self.lut[m, n, self.codeword[j, m, n]]

        if out_scale:
            out_mat = out_mat * out_scale.reshape(1, self.d_out)
        return out_mat

    def compute_fp_gemm(self, x: np.ndarray):
        # x: (bsz, d)
        # weight_mat: (out_dim, d)
        return x @ self.weight_mat.T



AQ = (128,1,256,4)
PQ = (128,4,256,1)

D, g = 1024, 16
AQLM = (D, D//g, 2, 256)
vq = VQ(AQLM, d_out=D)

D, B, g = 1024, 3, 4
TMAC = (D, D//g, B, int(2**g))
vq = VQ(TMAC, d_out=D, vq_type='TMAC')

inp = np.random.randn(2, D).astype(np.float16)
out_lut_gemv = vq.compute_lut_gemm(inp)
out_fp_gemm = vq.compute_fp_gemm(inp.astype(np.float32))
error = np.abs(out_lut_gemv - out_fp_gemm).mean()

print(out_lut_gemv)
print(out_fp_gemm)
print(error)

