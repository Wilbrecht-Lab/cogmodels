import numpy as np
import pandas as pd
from scipy.special import expit
import scipy
from numbers import Number
from abc import abstractmethod
from cogmodels.utils import Probswitch_2ABT, add_switch
from typing import List, Dict, Tuple, Any
import logging


class CogParam:
    def __init__(self, value=None, fixed_init: bool = False, n_id: int = 1, lb=None, ub=None):
        self.value = value
        self.fixed_init = fixed_init
        self.bounds = (lb, ub)
        self.n_id = n_id

    def eval(self):
        """
        Evaluates the parameter value.
        Returns:
            np.ndarray: An array of parameter values.
        Raises:
            ValueError: If the value is not a number or does not have an rvs method.
        """
        if isinstance(self.value, Number):
            return np.full(self.n_id, self.value)
        elif hasattr(self.value, "rvs"):
            return self.value.rvs(size=self.n_id)
        else:
            raise ValueError("Invalid parameter value: expected a number or a distribution with 'rvs'")


class CogModel:
    """
    Base model class for cognitive models in decision making. CogModel class is built with
    the goal to generalize to various task structures, while operating on a preset data format. 

    Parameters:
    ----------
    data: pd.DataFrame
        .columns: Subject, Decision, Reward, Target, *State* (applies to tasks with
        distinct sensory/reward states)
    """

    data_cols = [
        "ID",
        "Subject",
        "Session",
        "Trial",
        "blockTrial",
        "blockNum",
        "blockLength",
        "Target",
        "Decision",
        "Switch",
        "Reward",
        "Condition",
    ]

    # defines experiment columns that have little to no dependency to subject behavior
    expr_cols = [
        "Condition",
    ]

    def __init__(self):
        self.fixed_params = {}
        self.param_dict = {}
        self.fitted_params = None
        self.latent_names = []  # E.g. ['qdiff', 'rpe']
        self.summary = {}
        pass

    def create_params(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create a dataframe of evaluated parameters for each subject.
        Args:
            data (pd.DataFrame): Input data with an 'ID' column.
        Returns:
            pd.DataFrame: Dataframe containing parameters for each unique subject ID.
        """
        uniq_ids = data["ID"].unique()
        n_id = len(uniq_ids)
        new_params = {}
        for p in self.param_dict:
            self.param_dict[p].n_id = n_id
            new_params[p] = self.param_dict[p].eval()
        new_params["ID"] = uniq_ids
        return pd.DataFrame(new_params)

    @abstractmethod
    def latents_init(self, N:int):
        """
        Initialize the latent variables for simulation.
        Args:
            N (int): Total number of trials.
        Returns:
            tuple: Initialized latent variables.
        """    
        raise NotImplementedError()

    @abstractmethod
    def id_param_init(self, params:pd.DataFrame, param_id:str) -> Dict:
        """
        Initialize subject-specific parameters.
        Args:
            params (pd.DataFrame): Dataframe of parameters evaluated for all IDs.
            param_id: The ID to fetch parameters for.
        Returns:
            dict: A dictionary of parameters specific to the IDs.
        """
        raise NotImplementedError()

    @abstractmethod
    def fit_marginal(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit a marginal model on the data by adding additional variables.
        Args:
            data (pd.DataFrame): The input dataframe.
        Returns:
            pd.DataFrame: Dataframe with additional marginal columns.
        """
        raise NotImplementedError()


    def sim(self, data: pd.DataFrame, params: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        Simulate the model for the given data and parameters.
        This method should simulate trial-by-trial behavior; detailed implementation is task-specific.
        Args:
            data (pd.DataFrame): Experimental data.
            params (pd.DataFrame): Parameters per ID.
        Returns:
            pd.DataFrame: Dataframe with added latent variables.
        """
        raise NotImplementedError()

    def fit(self, data: pd.DataFrame, *args, **kwargs):
        """
        Fit the model parameters using maximum likelihood estimation.
        Returns:
            self: Fitted model object with updated parameters and summary.
        """
        data = self.fit_marginal(data)
        params_df = self.create_params(data)
        id_list = params_df["ID"]
        param_names = [c for c in params_df.columns if c != "ID"]
        params2x = lambda df: df.drop(columns="ID").values.ravel(
            order="C"
        )
        x2params = (
            lambda x: pd.DataFrame(
                x.reshape((-1, len(self.param_dict)), order="C"),
                columns=param_names,
                index=id_list,
            )
            .reset_index()
            .rename({"index": "ID"})
        )
        x0 = params2x(params_df)

        def nll(x:np.array) -> float:
            params = x2params(x)
            data_sim = self.sim(data, params, *args, **kwargs)
            # balanced weights
            # vcs = data_sim['Decision'].value_counts()
            # total1s, total0s = 0, 0
            # if 1 in vcs.index:
            #     total1s = vcs[1]
            # if 0 in vcs.index:
            #     total0s = vcs[0]
            # data_sim.loc[data_sim['Decision'] == 1, 'class_weight'] = total1s / (total1s + total0s)
            # data_sim.loc[data_sim['Decision'] == 0, 'class_weight'] = total0s / (total1s + total0s)
            # weight by class weights
            c_valid_sel = data_sim["Decision"] != -1
            p_s = self.get_proba(data_sim, params)[c_valid_sel].values

            c_vs = data_sim.loc[c_valid_sel, "Decision"].values
            epsilon = 1e-15 
            p_s = np.minimum(np.maximum(epsilon, p_s), 1 - epsilon)
            # try out class_weights
            return -(c_vs @ np.log(p_s) + (1 - c_vs) @ np.log(1 - p_s))

        params_bounds = [self.param_dict[pn].bounds for pn in param_names] * len(
            id_list
        )
        method = kwargs.get("method", "L-BFGS-B")

        res = scipy.optimize.minimize(
            nll, x0, method=method, bounds=params_bounds, tol=1e-6
        )
        if not res.success:
            logging.warning("failed", res.message)
        # perhaps if result gives failure, you throw an error
        self.fitted_params = x2params(res.x)
        negloglik = res.fun
        bic = len(res.x) * np.log(len(data)) + 2 * negloglik
        aic = 2 * len(res.x) + 2 * negloglik

        data_sim_opt = self.sim(data, self.fitted_params, *args, **kwargs)
        data["choice_p"] = self.get_proba(data_sim_opt)

        self.summary = {
            "bic": bic,
            "aic": aic,
            "latents": data[self.latent_names + ["choice_p"]].reset_index(drop=True),
        }
        return self

    def score(self, data: pd.DataFrame):
        """
        Cross validate the model on held-out data.
        """
        raise NotImplementedError()

    def emulate(self, *args, **kwargs):  
        """
        Emulate the model output; implementation depends on specific model.
        """
        raise NotImplementedError()


class CogModel2ABT_BWQ(CogModel):
    """
    Base class for 2-Alternative Forced Choice (2AFC/2ABT) cognitive models.
    input data must have the following columns:
    ID, Subject, Session, Trial, blockTrial, blockLength, Target, Decision, Switch, Reward, Condition
    """

    def __init__(self):
        super().__init__()
        self.k_action = 2
        self.fixed_params = {
            "predict_thres": 0.5,  # threshold for prediction
            "CF": False, # counterfactural RPE
            "sim_marg": False,  # whether or not to simulate marginal
        }  # Whether or not to calculate counterfactual rpe
        self.marg_name = "stay"

    
    def latents_init(self, N: int):
        """
        Initialize latent variables for simulation.
        Args:
            N (int): Number of trials.
        Returns:
            tuple: (qdiff, rpe, bias_array, weight_array)
        """
        qdiff = np.zeros(N, dtype=float)
        if self.fixed_params["CF"]:
            rpe = np.zeros((N, 2), dtype=float)
        else:
            rpe = np.zeros((N, 1), dtype=float)
        b_arr = np.zeros(N, dtype=float)
        w_arr = np.zeros((N, 2), dtype=float)
        return qdiff, rpe, b_arr, w_arr

    @abstractmethod
    def calc_q(self, b, w) -> np.ndarray:
        """
        Calculate action values from the given latents.
        Returns: 
            np.array of q action values
        """
        raise NotImplementedError()

    @abstractmethod
    def update_b(self, b, w, c_t, r_t, params_i):
        """
        abstract method for updating model latent b (belief)
        Parameters:
            b (float): current value of b, belief latent
            w (float): current value of w, weight latent    
            c_t (float): current choice
            r_t (float): current reward
            params_i (dict): current parameter values
        Returns: updated b, scalar or np.array
        """
        raise NotImplementedError()

    @abstractmethod
    def update_w(self, b, w, c_t, rpe_t, params_i):
        """
        abstract method for updating model latent w
        Parameters:
        b (float): current value of b, belief latent
        w (float): current value of w, weight latent    
        c_t (float): current choice
        r_t (float): current reward
        params_i (dict): current parameter values
        Returns: updated w, scalar or np.array
        """
        raise NotImplementedError()

    @abstractmethod
    def assign_latents(self, data: pd.DataFrame, qdiff, rpe, b_arr, w_arr):
        """
        Abstract method for saving simulated latents,
        Returns: data appended with columns storing different latents
        """
        pass

    def get_proba(self, data, params=None):
        """
        Calculate the choice probability for given data and parameters.
        Parameters:
            data (pd.DataFrame): Dataframe containing model simulation columns.
            params (pd.DataFrame, optional): Parameter dataframe; if None,
                uses fitted parameters.
        Returns:
            np.ndarray: Array of choice probabilities.
        """
        if "loglik" in data.columns:
            return np.exp(data["loglik"].values)
        else:
            if "beta" in self.fixed_params:  # change emulate function
                params_in = (
                    self.fitted_params[["ID", "st"]].copy()
                    if params is None
                    else params
                )
                params_in["beta"] = self.fixed_params["beta"]
            else:
                params_in = (
                    self.fitted_params[["ID", "beta", "st"]]
                    if params is None
                    else params
                )
            new_data = data.merge(params_in, how="left", on="ID")
            return expit(
                new_data["qdiff"] * new_data["beta"] + new_data["m"] * new_data["st"]
            )

    def select_action(self, qdiff, m_1back, params: Dict):
        """
        Select an action based on the softmax probability.
        Params:
            qdiff: Difference in Q-values.
            m_1back: Value from marginal variable.
            params (dict): Parameter dictionary containing 'beta' and 'st'.
        Returns:
            int: The selected action.
        """
        choice_p = expit(qdiff * params["beta"] + m_1back * params["st"])
        return int(np.random.random() <= choice_p)

    def marginal_init(self):
        """
        Initialize the marginal latent.
        """
        return 0

    def update_m(self, c_t: int, m: float, params: Dict):
        """
        Update the marginal variable based on current choice.
        Params:
            c_t: Current decision (0, 1, or -1 for miss trial).
            m: Previous marginal estimate.
            params (dict): Parameters (not used in current formulation).
        Returns:
            int: Updated marginal latent m.

        """
        m1 = 1
        if c_t == 0:
            m1 = -1
        elif c_t == -1:
            m1 = 0
        assert not np.isnan(c_t), "Choice value cannot be NaN"
        return m1

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict actions based on choice probabilities.
        Returns:
            np.ndarray: Array of predicted actions.
        """
        return (self.get_proba(data) >= self.fixed_params["predict_thres"]).astype(
            float
        )

    def fit_marginal(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fits a simple marginal latent, estimating choice marginal distribution, based on previous choice, 
        useful for BI and RL model
        Args:
            data (pd.DataFrame): Input data.
        Returns:
            pd.DataFrame: Data with a new column 'm'.
        """
        # test marginal stay
        if self.fixed_params["sim_marg"]:
            return data
        c = data["Decision"]
        c_lag = c.shift(1)
        data["m"] = -1 # default marginal value
        data.loc[c_lag == 1, "m"] = 1
        data.loc[data["Trial"] == 1, "m"] = 0
        data.loc[c_lag == -1, "m"] = 0
        return data

    def sim(self, data, params, *args, **kwargs):
        """
        Simulates the model for single ID that matches the data

        Pseudo code:
        ```
        def sim(data, params):
            # b: belief related latents
            # w: learning weight of models
            *latents <- latents_init # initialize all model latents
            rs, cs <- extract_data(data) # load reward and choice data
            for i=1:len(data):
                if new subject:
                    params_d <- id_param_init(params, get_data_id(data, i))
                    # initialize when there is a new subject ID
                r, c = rs[i], cs[i]
                qs <- calc_qs(b, w) # calculate q values given
                rpe <- calculate_rpe(r, qs)
                w <- update_w(b, w, c, rpe, params_d)
                b <- update_b(b, w, c, rpe, params_d)
            return assign_latents(data, w,b,rpe ...)
        ```

        Input:
            data: pd.DataFrame
            ... params listed in class description
            params: pd.DataFrame
            ... containing parameters of interest

        Returns:
            pd.DataFrame: data with new columns filled with latents listed in class description
        """
        N = data.shape[0]
        qdiff, rpe, b_arr, w_arr = self.latents_init(N)

        c = data["Decision"]
        sess = data["Session"]
        subj = data["Subject"]
        id_i = data["ID"]
        r = data["Reward"]

        params_d = self.id_param_init(params, id_i.iat[0])
        b0, w0, gam = params_d["b0"], params_d["w0"], params_d["gam"]
        b, w = b0, w0
        if self.fixed_params["sim_marg"]:
            margs = np.zeros(N)
            m1back = self.marginal_init()

        # np.seterr(all='raise')
        for n in range(N):
            # Reset parameters when ID/subject changes or session changes
            if (n > 0) and (id_i.iat[n] != id_i.iat[n - 1]):
                params_d = self.id_param_init(params, id_i.iat[n])

            if (n == 0) or (subj.iat[n] != subj.iat[n - 1]):
                b = b0
                w = np.copy(w0)
                if self.fixed_params["sim_marg"]:
                    m1back = self.marginal_init()
            elif sess.iat[n] != sess.iat[n - 1]:
                b = b0
                w = w0 * gam + (1 - gam) * w
                if self.fixed_params["sim_marg"]:
                    m1back = self.marginal_init()
            ## Action value calculation
            qs = self.calc_q(b, w).ravel()
            # compute value difference
            qdiff[n] = qs[1] - qs[0]
            if self.fixed_params["sim_marg"]:
                margs[n] = m1back
            ## Model update
            if c.iat[n] == -1:
                # handling miss trials
                rpe[n, :] = np.nan
                w_arr[n, :] = w
                b_arr[n] = b
            else:
                rpe_c = r.iat[n] - qs[c.iat[n]]
                if self.fixed_params["CF"]:
                    rpe_cf = (1 - r.iat[n]) - qs[1 - c.iat[n]]
                    rpe_t = np.array([rpe_c, rpe_cf])
                else:
                    rpe_t = rpe_c
                rpe[n, :] = rpe_t
                # w, b reflects information prior to reward
                w_arr[n, :] = w
                b_arr[n] = b
                w = self.update_w(b, w, c.iat[n], rpe_t, params_d)
                # Updating b!
                b = self.update_b(b, w, c.iat[n], r.iat[n], params_d)
            if self.fixed_params["sim_marg"]:
                m1back = self.update_m(c.iat[n], m1back, params_d)

        if self.fixed_params["sim_marg"]:
            data["m"] = margs

        data = self.assign_latents(data, qdiff, rpe, b_arr, w_arr)
        # qdiff[n], rpe[n], b, w, b_arr[n], w_arr[n]
        return data

    def generate(self, params, *args, **kwargs):
        """
        Generate synthetic data based on parameters for single ID.
        Args:
            params (pd.DataFrame): Dataframe containing one ID's parameters and simulation settings.
            ID, vars, n_trial, n_session
            ... containing parameters of interest
        Returns:
            pd.DataFrame: Generated data with simulation latents.
        """
        uniq_id = params["ID"].values[0]
        n_trial = params["n_trial"].values[0]
        n_session = params["n_session"].values[0]
        N = n_trial * n_session
        qdiff, rpe, b_arr, w_arr = self.latents_init(N)
        c = np.zeros(N, dtype=int)
        r = np.zeros(N, dtype=int)

        data = {k: np.zeros(N, dtype=int) for k in ["Target", "Decision", "Reward"]}
        data["Session"] = np.repeat(
            [f"{i:02d}" for i in range(1, n_session + 1)], n_trial
        )
        data["Trial"] = np.tile([i + 1 for i in range(n_trial)], n_session)
        data = pd.DataFrame(data)
        data["Subject"] = uniq_id
        data["ID"] = uniq_id
        sess = data["Session"]

        task_params = {"blockLen_range": (8, 15), "condition": "75-0"}
        task_params.update(kwargs)
        probs = task_params["condition"].split("-")
        data["Condition"] = task_params["condition"]
        p_cor, p_inc = [float(p) / 100 for p in probs]
        task_params.update({"p_cor": p_cor, "p_inc": p_inc})
        task = Probswitch_2ABT(**task_params)

        # fix_cols = ['ID', 'Subject', 'Session', 'Trial', 'blockTrial', 'blockLength', 'Target', 'Condition']
        # ID, Subject, Session, Trial, blockTrial, blockLength, Target, Decision, Switch, Reward, Condition
        params_d = self.id_param_init(params, uniq_id)
        b0, w0, gam = params_d["b0"], params_d["w0"], params_d["gam"]

        b, w = b0, w0
        m_1back = self.marginal_init()

        # when ID changes: certain parameter changes
        # np.seterr(all='raise')
        for n in range(N):
            if (n != 0) and (sess.iat[n] != sess.iat[n - 1]):
                b = b0
                w = w0 * gam + (1 - gam) * w
                m_1back = self.marginal_init()
                task.initialize()
            ## Action value calculation
            qs = self.calc_q(b, w).ravel()
            # compute value difference
            qdiff[n] = qs[1] - qs[0]

            c_t = self.select_action(qdiff[n], m_1back, params_d)
            m_1back = self.update_m(c_t, m_1back, params_d)

            data.loc[n, "Target"] = task.target
            data.loc[n, "blockTrial"] = task.btrial
            data.loc[n, "blockNum"] = task.blockN
            r_t = task.getOutcome(c_t)

            ## Model update
            rpe_c = r_t - qs[c_t]

            if self.fixed_params["CF"]:
                rpe_cf = (1 - r_t) - qs[1 - c_t]
                rpe_t = np.array([rpe_c, rpe_cf])
            else:
                rpe_t = rpe_c
            c[n], r[n] = c_t, r_t
            rpe[n, :] = rpe_t
            # w, b reflects information prior to reward
            w_arr[n, :] = w
            b_arr[n] = b
            w = self.update_w(b, w, c[n], rpe_t, params_d)
            # Updating b!
            b = self.update_b(b, w, c[n], r[n], params_d)

        data["Decision"] = c
        data["Reward"] = r
        v = data.groupby(["ID", "Session", "blockNum"], as_index=False).apply(len)
        v.columns = list(v.columns[:3]) + ["blockLength"]
        data = data.merge(v, how="left", on=["ID", "Session", "blockNum"])
        data = add_switch(data)
        return data

    def emulate(self, data: pd.DataFrame, params, *args, **kwargs) -> pd.DataFrame:
        """
        Emulate model behavior for the input data and parameter set.
        Args:
            data (pd.DataFrame): Experimental data.
            params (pd.DataFrame): Subject-specific model parameters.
        Returns:
            pd.DataFrame: Data with emulated decision and reward latents.
        """
        N = data.shape[0]
        qdiff, rpe, b_arr, w_arr = self.latents_init(N)
        c = np.zeros(N, dtype=int)
        r = np.zeros(N, dtype=int)

        sess = data["Session"]
        subj = data["Subject"]
        id_i = data["ID"]
        targets = data["Target"]
        c_data = data["Decision"]
        r_data = data["Reward"]

        probs = data["Condition"].unique()[0].split("-")
        p_cor, p_inc = [float(p) / 100 for p in probs]
        fix_cols = [
            "ID",
            "Subject",
            "Session",
            "Trial",
            "blockTrial",
            "blockLength",
            "Target",
            "Condition",
        ]
        emu_data = data[fix_cols].reset_index(drop=True)
        params_d = self.id_param_init(params, id_i.iat[0])
        b0, w0, gam = params_d["b0"], params_d["w0"], params_d["gam"]

        b, w = b0, w0
        m_1back = self.marginal_init()

        # when ID changes: certain parameter changes
        # np.seterr(all='raise')
        for n in range(N):
            # initializing latents
            if (n == 0) or (id_i.iat[n] != id_i.iat[n - 1]):
                params_d = self.id_param_init(params, id_i.iat[n])
                # generalize recency

            if (n == 0) or (subj.iat[n] != subj.iat[n - 1]):
                b = b0
                w = np.copy(w0)
                m_1back = self.marginal_init()
            elif sess.iat[n] != sess.iat[n - 1]:
                b = b0
                w = w0 * gam + (1 - gam) * w
                m_1back = self.marginal_init()
            ## Action value calculation
            qs = self.calc_q(b, w).ravel()
            # compute value difference
            qdiff[n] = qs[1] - qs[0]

            c_t = self.select_action(qdiff[n], m_1back, params_d)
            m_1back = self.update_m(c_t, m_1back, params_d)

            if c_t == c_data.iat[n]:
                r_t = r_data.iat[n]
            else:
                dice = np.random.random()
                if targets.iat[n] == c_t:
                    r_t = int(dice <= p_cor)
                else:
                    r_t = int(dice <= p_inc)  # check this block
            ## Model update
            rpe_c = r_t - qs[c_t]

            if self.fixed_params["CF"]:
                rpe_cf = (1 - r_t) - qs[1 - c_t]
                rpe_t = np.array([rpe_c, rpe_cf])
            else:
                rpe_t = rpe_c
            c[n], r[n] = c_t, r_t
            rpe[n, :] = rpe_t
            # w, b reflects information prior to reward
            w_arr[n, :] = w
            b_arr[n] = b
            w = self.update_w(b, w, c[n], rpe_t, params_d)
            # Updating b!
            b = self.update_b(b, w, c[n], r[n], params_d)
        emu_data["Decision"] = c
        emu_data["Reward"] = r
        emu_data = add_switch(emu_data)
        emu_data = self.assign_latents(emu_data, qdiff, rpe, b_arr, w_arr)
        # qdiff[n], rpe[n], b, w, b_arr[n], w_arr[n]
        return emu_data