from cogmodels.utils_test import *
from cogmodels import *

if __name__ == "__main__":
    import time
    for model in [
        # RL_Forgetting,
        RL_Forgetting3p,
        # # RL_FQST,
        # WSLS,
        Pearce_Hall,
        # RL_Grossman,
        # # RL_Grossman_prime,
        # RL_Grossman_nof,
        # RL_Grossman_nost,
        BRL_wrp,
        # BRL_fwr,
        # # BRL_fw,
        # # BRL_fp,
        # BRL_wr,
        # RL_4p,
        # BIModel_fixp,
        # BIModel,
        # RFLR,
        # RLCF,
        # # PCModel_fixpswgam,
        # PCBRL,
    ]:  # BRL_fp, BRL_wr, RL_4p, RLCF, PCModel_fixpswgam, PCBRL
        # for model in [BIModel_fixp, PCModel_fixpswgam, BI_log, PCBRL, RL_4p]:
        print(str(model()))
        # test_mp_multiple_sessions(model)
        test_mp_multiple_animals(model)
        test_model_genrec_BSD(model)