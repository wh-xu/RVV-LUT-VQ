import pandas as pd
import numpy as np
from vq import VQ


df = pd.DataFrame(columns=[
    "Dataflow", 
    "VQ_type",
    "Din",
    "Dout",
    "VLEN",
    "LMUL",
    "BW",
    "D",
    "M",
    "N",
    "K",
    "N_tile",
    "M_tile",
    "CW_tile",
    "Offchip_access_codeword_MB", 
    "Offchip_access_codebook_MB", 
    "Offchip_access_lut_MB", 
    "Offchip_access_inp_MB",
    "Offchip_access_psum_MB",
    ]
)


for dataflow in ['OMND', 'ODMN', 'DMNO', 'MNOD', 'MNDO', 'DOMN', 'OMND-compact', 'ODMN-compact', 'DMNO-compact', 'MNOD-compact', 'MNDO-compact', 'DOMN-compact', 'VeLU']:
    # for cfg in ["TMAC_GEMV", "PQ", "AQLM_GEMV", "TMAC_GEMM", "PQ_Batch", "AQLM_GEMM"]:
    for cfg in ["TMAC_GEMV", "PQ", "RQ", "AQLM_GEMV"]:
        if cfg == "TMAC_GEMV":
            D, B, g = 4096, 4, 4
            Din, Dout = 1, D
            TMAC = (D, D//g, B, int(2**g))
            vq = VQ(TMAC, d_out=Dout, vq_type='TMAC')
        elif cfg == "PQ":
            N, D, m = 1024, 128, 8
            Din, Dout = 1, N
            PQ = (D, m, 1, 256)
            vq = VQ(PQ, d_out=Dout)
        elif cfg == "RQ":
            N, D, m = 1024, 128, 1
            Din, Dout = 1, N
            RQ = (D, m, 4, 256)
            vq = VQ(RQ, d_out=Dout)
        elif cfg == "AQLM_GEMV":
            D, g = 4096, 16
            Din, Dout = 1, D
            AQLM = (D, D//g, 2, 256)
            vq = VQ(AQLM, d_out=Dout)
        elif cfg == "TMAC_GEMM":
            D, B, g = 4096, 4, 4
            Din, Dout = D, D
            TMAC = (D, D//g, B, int(2**g))
            vq = VQ(TMAC, d_out=Dout, vq_type='TMAC')
        elif cfg == "PQ_Batch":
            N, D, m = 1024, 128, 8
            Din, Dout = 64, N
            PQ = (D, m, 1, 256)
            vq = VQ(PQ, d_out=Dout)
        elif cfg == "AQLM_GEMM":
            D, g = 4096, 16
            Din, Dout = D, D
            AQLM = (D, D//g, 2, 256)
            vq = VQ(AQLM, d_out=Dout)
        else:
            raise ValueError(f"Invalid configuration: {cfg}")
        
        inp = np.random.randn(Din, D).astype(np.float16)
        out_fp_gemm = vq.compute_fp_gemm(inp.astype(np.float32))
        out_lut_gemv = vq.dataflow_sim(inp, dataflow)
        error = np.abs(out_lut_gemv - out_fp_gemm).mean()

        if vq.perf_cnt["offchip_access_codeword"] > 0:
            print(f"\nDataflow: {dataflow}, Error: {error}")
            print(vq.perf_cnt)

            df.loc[len(df)] = {
                "Dataflow": dataflow,
                "VQ_type": cfg,
                "Din": Din,
                "Dout": Dout,
                "D": D,
                "M": vq.n_subvec,
                "N": vq.n_codebook,
                "K": vq.n_cluster,
                "VLEN": vq.VLEN,
                "LMUL": vq.LMUL,
                "BW": vq.BW,
                "N_tile": vq.n_codebook_tile,
                "M_tile": vq.n_subvec_tile,
                "CW_tile": vq.n_cw_tile,
                "Offchip_access_codeword_MB": vq.perf_cnt["offchip_access_codeword"] / 1024**2/8,
                "Offchip_access_codebook_MB": vq.perf_cnt["offchip_access_codebook"] / 1024**2/8,
                "Offchip_access_lut_MB": vq.perf_cnt["offchip_access_lut"] / 1024**2/8,
                "Offchip_access_inp_MB": vq.perf_cnt["offchip_access_inp"] / 1024**2/8,
                "Offchip_access_psum_MB": vq.perf_cnt["offchip_access_psum"] / 1024**2/8,
            }
            print(df)

df["Offchip_access_total_MB"] = df["Offchip_access_codeword_MB"] + df["Offchip_access_codebook_MB"] + df["Offchip_access_lut_MB"] + df["Offchip_access_inp_MB"] + df["Offchip_access_psum_MB"]

df.sort_values(by=["VQ_type", "Dataflow"], ascending=False, inplace=True)

df.to_csv("dataflow_sim.csv", index=False)

