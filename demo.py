from cogmodels.utils_test import *
from cogmodels import *

if __name__ == "__main__":
    # test_mp_multiple_animals(PCModel, 'PC')
    # test_mp_multiple_animals(PCModel_fixpswgam, 'PC_fixpswgam')
    # test_mp_multiple_animals(BIModel, 'BI')
    import time

    # test_model_genrec_eckstein2022()
    # test_model_genrec_eckstein2022_RL()
    # test_model_genrec_eckstein2022()
    # test_model_eckstein2022_RLCF()
    # test_model_eckstein2022_RLCF()
    print("rlmeta")
    # models = [RL_Forgetting3p, RL_Grossman, Pearce_Hall, RL_4p, RLCF, RFLR, BRL_fwr, BIModel_fixp]
    # test_model_identifiability_mp(models, f"{DATA_ARG}_3s5ht")
    # gen_arg = f"{DATA_ARG}_3s5ht"
    # model = RL_Grossman
    # method = "L-BFGS-B"

    # gendata = pd.read_csv(
    #     os.path.join(
    #         CACHE_FOLDER, f"genrec_{gen_arg}_{str(model())}_{method}_gendata.csv"
    #     )
    # )
    # fit_model_all_subjects(
    #     gendata.iloc[: len(gendata) // 20].reset_index(drop=True), RL_Grossman
    # )

    # test_model_genrec_eckstein2022_RLCF()
    # test_model_genrec_eckstein2022()
    # test_model_genrec_eckstein2022_BIfp()
    # test_model_genrec_eckstein2022_PCf()
    for model in [
        # RL_Forgetting,
        # RL_Forgetting3p,
        # # RL_FQST,
        # WSLS,
        # Pearce_Hall,
        RL_Grossman,
        # # RL_Grossman_prime,
        # RL_Grossman_nof,
        # RL_Grossman_nost,
        # BRL_wrp,
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