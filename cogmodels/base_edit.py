"""
Summary of Improvements
Documentation & Type Hints: Added clearer docstrings and type hints for better developer understanding.
Naming Conventions: Changed ambiguous variable names (e.g., “qdiff” → “q_diff”, “id_i” → “subject_ids”) to be more descriptive.
Error Handling: Instead of returning error objects, errors (e.g., in parameter evaluation and optimization failure) are now raised.
Code Cleanup: Removed unused or outdated commented code and ensured consistent formatting throughout.
These changes should make the codebase more maintainable, self-documenting, and robust for future development.

Analysis
CogParam.eval Method

Issue: The method returns a ValueError instance instead of raising one when the value isn’t a Number or does not have an rvs method.
Improvement: Raise a ValueError with a descriptive message.
Inconsistent Naming and Docstrings

Issue: Some methods (e.g., sim, id_param_init, latents_init) have minimal or no documentation. Variable names (e.g., “c”, “qdiff”, “id_i”) could be more descriptive.
Improvement: Add type hints and more detailed docstrings, and rename ambiguous variables (e.g., “id_i” to “subject_ids”).
Optimization in Fit Methods

Issue: In the fit methods, the optimization failure is only printed.
Improvement: Instead of just printing a failure message, raise an exception so that higher-level code can handle it properly.
Docstring Formatting and Comments

Issue: Some comments are outdated or commented-out code if not needed.
Improvement: Remove redundancy and clean up comments.
Potential Runtime Optimizations

Issue: There are iterative loops over DataFrame rows in the simulation methods.
Improvement: While vectorization might improve performance, readability and handling subject/session breaks are critical. Consider future refactoring if performance becomes an issue.
"""

import numpy as np
import pandas as pd
from scipy.special import expit
import scipy
from numbers import Number
from abc import abstractmethod
from cogmodels.utils import Probswitch_2ABT, add_switch


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
    Base model class for cognitive models in decision making.
    
    Parameters:
        data (pd.DataFrame): Data with expected columns including Subject, Decision,
            Reward, Target, and optionally State.
    """
    data_cols = [
        "ID", "Subject", "Session", "Trial", "blockTrial",
        "blockNum", "blockLength", "Target", "Decision",
        "Switch", "Reward", "Condition"
    ]

    # Additional experiment-specific columns
    expr_cols = [
        "ID", "Subject", "Session", "Trial", "blockTrial",
        "blockNum", "blockLength", "Target", "Reward", "Condition"
    ]

    def __init__(self):
        self.fixed_params = {}
        self.param_dict = {}
        self.fitted_params = None
        self.latent_names = []  # E.g., ['qdiff', 'rpe']
        self.summary = {}

    def create_params(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create a dataframe of evaluated parameters for each subject.

        Args:
            data (pd.DataFrame): Input data with an 'ID' column.

        Returns:
            pd.DataFrame: Dataframe containing parameters for each unique subject ID.
        """
        unique_ids = data["ID"].unique()
        n_ids = len(unique_ids)
        new_params = {}
        for param_key in self.param_dict:
            self.param_dict[param_key].n_id = n_ids
            new_params[param_key] = self.param_dict[param_key].eval()
        new_params["ID"] = unique_ids
        return pd.DataFrame(new_params)

    @abstractmethod
    def latents_init(self, N: int):
        """
        Initialize the latent variables for simulation.

        Args:
            N (int): Total number of trials.

        Returns:
            tuple: Initialized latent variables.
        """
        pass

    @abstractmethod
    def id_param_init(self, params: pd.DataFrame, subject_id):
        """
        Initialize subject-specific parameters.

        Args:
            params (pd.DataFrame): Dataframe of parameters evaluated for all subjects.
            subject_id: The subject ID to fetch parameters for.

        Returns:
            dict: A dictionary of parameters specific to the subject.
        """
        pass

    @abstractmethod
    def fit_marginal(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit a marginal model on the data by adding additional variables.
        
        Args:
            data (pd.DataFrame): The input dataframe.
        
        Returns:
            pd.DataFrame: Dataframe with additional marginal columns.
        """
        return data

    def sim(self, data: pd.DataFrame, params: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        Simulate the model for the given data and parameters.
        
        This method should simulate trial-by-trial behavior; detailed implementation is task-specific.

        Args:
            data (pd.DataFrame): Experimental data.
            params (pd.DataFrame): Parameters per subject.
        
        Returns:
            pd.DataFrame: Dataframe with added latent variables.
        """
        pass

    def fit(self, data: pd.DataFrame, *args, **kwargs):
        """
        Fit the model parameters using maximum likelihood estimation.
        
        Returns:
            self: Fitted model object with updated parameters and summary.
        
        Raises:
            RuntimeError: If the optimization fails.
        """
        data = self.fit_marginal(data)
        params_df = self.create_params(data)
        subject_ids = params_df["ID"]
        param_names = [c for c in params_df.columns if c != "ID"]

        params2x = lambda df: df.drop(columns="ID").values.ravel(order="C")
        x2params = lambda x: pd.DataFrame(
            x.reshape((-1, len(self.param_dict)), order="C"),
            columns=param_names,
            index=subject_ids
        ).reset_index().rename(columns={"index": "ID"})

        x0 = params2x(params_df)

        def neg_log_likelihood(x: np.ndarray) -> float:
            current_params = x2params(x)
            data_sim = self.sim(data.copy(), current_params, *args, **kwargs)
            valid_mask = data_sim["Decision"] != -1
            prob_sim = self.get_proba(data_sim, current_params)[valid_mask].values
            decisions = data_sim.loc[valid_mask, "Decision"].values
            epsilon = 1e-15
            prob_sim = np.clip(prob_sim, epsilon, 1 - epsilon)
            return -(decisions @ np.log(prob_sim) + (1 - decisions) @ np.log(1 - prob_sim))

        params_bounds = [self.param_dict[pn].bounds for pn in param_names] * len(subject_ids)
        method = kwargs.get("method", "L-BFGS-B")

        res = scipy.optimize.minimize(neg_log_likelihood, x0, method=method, bounds=params_bounds, tol=1e-6)
        if not res.success:
            raise RuntimeError(f"Optimization failed: {res.message}")

        self.fitted_params = x2params(res.x)
        negloglik = res.fun
        bic = len(res.x) * np.log(len(data)) + 2 * negloglik
        aic = 2 * len(res.x) + 2 * negloglik

        data_sim_opt = self.sim(data.copy(), self.fitted_params, *args, **kwargs)
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
        pass

    def emulate(self, *args, **kwargs):
        """
        Emulate the model output; implementation depends on specific model.
        """
        pass

    def get_proba(self, data: pd.DataFrame, params: pd.DataFrame = None) -> np.ndarray:
        """
        Calculate the choice probability for given data and parameters.
        
        Args:
            data (pd.DataFrame): Dataframe containing model simulation columns.
            params (pd.DataFrame, optional): Parameter dataframe; if None,
                uses fitted parameters.

        Returns:
            np.ndarray: Array of choice probabilities.
        """
        if "loglik" in data.columns:
            return np.exp(data["loglik"].values)
        else:
            if "beta" in self.fixed_params:  # using fixed beta if provided
                params_in = self.fitted_params[["ID", "st"]].copy() if params is None else params
                params_in["beta"] = self.fixed_params["beta"]
            else:
                params_in = self.fitted_params[["ID", "beta", "st"]] if params is None else params

            merged_data = data.merge(params_in, how="left", on="ID")
            return expit(merged_data["qdiff"] * merged_data["beta"] + merged_data["m"] * merged_data["st"])

    def select_action(self, q_diff: float, marginal_value: float, params: dict) -> int:
        """
        Select an action based on the softmax probability.

        Args:
            q_diff (float): Difference in Q-values.
            marginal_value (float): Value from marginal influence.
            params (dict): Parameter dictionary containing 'beta' and 'st'.

        Returns:
            int: The selected action.
        """
        choice_probability = expit(q_diff * params["beta"] + marginal_value * params["st"])
        return int(np.random.random() <= choice_probability)


class CogModel2ABT_BWQ(CogModel):
    """
    Base class for 2-Alternative Forced Choice (2AFC/2ABT) cognitive models.
    
    The input data must include the following columns:
    ID, Subject, Session, Trial, blockTrial, blockLength, Target, Decision, Switch, Reward, and Condition.
    """
    def __init__(self):
        super().__init__()
        self.k_action = 2
        self.fixed_params = {
            "predict_thres": 0.5,
            "CF": False,
            "sim_marg": False,
        }  # Whether or not to calculate counterfactual reward prediction error
        self.marg_name = "stay"

    def latents_init(self, N: int):
        """
        Initialize latent variables for simulation.

        Args:
            N (int): Number of trials.

        Returns:
            tuple: (qdiff, rpe, bias_array, weight_array)
        """
        q_diff = np.zeros(N, dtype=float)
        if self.fixed_params["CF"]:
            rpe = np.zeros((N, 2), dtype=float)
        else:
            rpe = np.zeros((N, 1), dtype=float)
        bias_array = np.zeros(N, dtype=float)
        weight_array = np.zeros((N, 2), dtype=float)
        return q_diff, rpe, bias_array, weight_array

    @abstractmethod
    def calc_q(self, bias, weight) -> np.ndarray:
        """
        Calculate action values from the given latents.

        Args:
            bias: Current bias state.
            weight: Current weight state.

        Returns:
            np.ndarray: Array of action values.
        """
        pass

    @abstractmethod
    def update_b(self, bias, weight, choice, reward, params: dict):
        """
        Update the belief (bias) latent variable.

        Args:
            bias: Current bias.
            weight: Current weight.
            choice: Current action.
            reward: Received reward.
            params (dict): Subject-specific parameters.

        Returns:
            Updated bias value.
        """
        pass

    @abstractmethod
    def update_w(self, bias, weight, choice, rpe, params: dict):
        """
        Update the weight latent variable.

        Args:
            bias: Current bias.
            weight: Current weight.
            choice: Current action.
            rpe: Reward prediction error.
            params (dict): Subject-specific parameters.

        Returns:
            Updated weight value.
        """
        pass

    @abstractmethod
    def assign_latents(self, data: pd.DataFrame, q_diff, rpe, bias_arr, weight_arr) -> pd.DataFrame:
        """
        Append latent variables to the data.

        Args:
            data (pd.DataFrame): Base data.
            q_diff, rpe, bias_arr, weight_arr: Simulated latent variables.

        Returns:
            pd.DataFrame: Data augmented with latent variable columns.
        """
        pass

    def marginal_init(self) -> int:
        """
        Initialize the marginal value.
        
        Returns:
            int: Initial marginal value.
        """
        return 0

    def update_m(self, choice: int, current_m: float, params: dict) -> int:
        """
        Update the marginal variable based on current choice.

        Args:
            choice (int): Current decision (0, 1, or -1 for miss trial).
            current_m (float): Previous marginal estimate.
            params (dict): Parameters (not used in current formulation).

        Returns:
            int: Updated marginal value.
        """
        new_m = 1
        if choice == 0:
            new_m = -1
        elif choice == -1:
            new_m = 0
        assert not np.isnan(choice), "Choice value cannot be NaN."
        return new_m

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict actions based on choice probabilities.
        
        Returns:
            pd.DataFrame: Column of predicted choices.
        """
        return (self.get_proba(data) >= self.fixed_params["predict_thres"]).astype(float)

    def fit_marginal(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit a simple marginal model based on previous trial information.
        
        Args:
            data (pd.DataFrame): Input data.
        
        Returns:
            pd.DataFrame: Data with a new column 'm'.
        """
        if self.fixed_params["sim_marg"]:
            return data
        choices = data["Decision"]
        lagged_choices = choices.shift(1)
        data["m"] = -1  # Default marginal value
        data.loc[lagged_choices == 1, "m"] = 1
        data.loc[data["Trial"] == 1, "m"] = 0
        data.loc[lagged_choices == -1, "m"] = 0
        return data

    def sim(self, data: pd.DataFrame, params: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        Simulate the 2ABT model on the data.

        Args:
            data (pd.DataFrame): Experimental data.
            params (pd.DataFrame): Subject-specific parameters.
        
        Returns:
            pd.DataFrame: Simulation data with appended latent variables.
        """
        N = data.shape[0]
        q_diff, rpe, bias_arr, weight_arr = self.latents_init(N)

        choices = data["Decision"]
        sessions = data["Session"]
        subjects = data["Subject"]
        subject_ids = data["ID"]
        rewards = data["Reward"]

        params_d = self.id_param_init(params, subject_ids.iat[0])
        bias_init, weight_init, gam = params_d["b0"], params_d["w0"], params_d["gam"]
        bias_current, weight_current = bias_init, weight_init
        if self.fixed_params["sim_marg"]:
            marginal_values = np.zeros(N)
            marginal_prev = self.marginal_init()

        for n in range(N):
            # Reset parameters when subject or session changes.
            if (n > 0) and (subject_ids.iat[n] != subject_ids.iat[n - 1]):
                params_d = self.id_param_init(params, subject_ids.iat[n])
            if (n == 0) or (subjects.iat[n] != subjects.iat[n - 1]):
                bias_current = bias_init
                weight_current = np.copy(weight_init)
                if self.fixed_params["sim_marg"]:
                    marginal_prev = self.marginal_init()
            elif sessions.iat[n] != sessions.iat[n - 1]:
                bias_current = bias_init
                weight_current = weight_init * gam + (1 - gam) * weight_current
                if self.fixed_params["sim_marg"]:
                    marginal_prev = self.marginal_init()

            qs = self.calc_q(bias_current, weight_current).ravel()
            q_diff[n] = qs[1] - qs[0]
            if self.fixed_params["sim_marg"]:
                marginal_values[n] = marginal_prev

            if choices.iat[n] == -1:
                rpe[n, :] = np.nan
                weight_arr[n, :] = weight_current
                bias_arr[n] = bias_current
            else:
                rpe_current = rewards.iat[n] - qs[choices.iat[n]]
                if self.fixed_params["CF"]:
                    rpe_counterfactual = (1 - rewards.iat[n]) - qs[1 - choices.iat[n]]
                    rpe_trial = np.array([rpe_current, rpe_counterfactual])
                else:
                    rpe_trial = rpe_current
                rpe[n, :] = rpe_trial
                weight_arr[n, :] = weight_current
                bias_arr[n] = bias_current
                weight_current = self.update_w(bias_current, weight_current, choices.iat[n], rpe_trial, params_d)
                bias_current = self.update_b(bias_current, weight_current, choices.iat[n], rewards.iat[n], params_d)
            if self.fixed_params["sim_marg"]:
                marginal_prev = self.update_m(choices.iat[n], marginal_prev, params_d)

        if self.fixed_params["sim_marg"]:
            data["m"] = marginal_values

        data = self.assign_latents(data, q_diff, rpe, bias_arr, weight_arr)
        return data

    def generate(self, params: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        Generate synthetic data based on parameters for one subject.
        
        Args:
            params (pd.DataFrame): Dataframe containing one subject's parameters and simulation settings.
        
        Returns:
            pd.DataFrame: Generated data with simulation latents.
        """
        unique_id = params["ID"].values[0]
        n_trial = params["n_trial"].values[0]
        n_session = params["n_session"].values[0]
        N = n_trial * n_session
        q_diff, rpe, bias_arr, weight_arr = self.latents_init(N)
        choices = np.zeros(N, dtype=int)
        rewards = np.zeros(N, dtype=int)

        data_dict = {k: np.zeros(N, dtype=int) for k in ["Target", "Decision", "Reward"]}
        data_dict["Session"] = np.repeat([f"{i:02d}" for i in range(1, n_session + 1)], n_trial)
        data_dict["Trial"] = np.tile(np.arange(1, n_trial + 1), n_session)
        data = pd.DataFrame(data_dict)
        data["Subject"] = unique_id
        data["ID"] = unique_id

        task_params = {"blockLen_range": (8, 15), "condition": "75-0"}
        task_params.update(kwargs)
        condition_str = task_params["condition"]
        data["Condition"] = condition_str
        p_cor, p_inc = [float(p) / 100 for p in condition_str.split("-")]
        task_params.update({"p_cor": p_cor, "p_inc": p_inc})
        task = Probswitch_2ABT(**task_params)

        params_d = self.id_param_init(params, unique_id)
        bias_init, weight_init, gam = params_d["b0"], params_d["w0"], params_d["gam"]
        bias_current, weight_current = bias_init, weight_init
        marginal_prev = self.marginal_init()

        for n in range(N):
            if n > 0 and data["Session"].iat[n] != data["Session"].iat[n - 1]:
                bias_current = bias_init
                weight_current = weight_init * gam + (1 - gam) * weight_current
                marginal_prev = self.marginal_init()
                task.initialize()

            qs = self.calc_q(bias_current, weight_current).ravel()
            q_diff[n] = qs[1] - qs[0]
            chosen_action = self.select_action(q_diff[n], marginal_prev, params_d)
            marginal_prev = self.update_m(chosen_action, marginal_prev, params_d)

            data.loc[n, "Target"] = task.target
            data.loc[n, "blockTrial"] = task.btrial
            data.loc[n, "blockNum"] = task.blockN
            reward_trial = task.getOutcome(chosen_action)

            rpe_current = reward_trial - qs[chosen_action]
            if self.fixed_params["CF"]:
                rpe_counterfactual = (1 - reward_trial) - qs[1 - chosen_action]
                rpe_trial = np.array([rpe_current, rpe_counterfactual])
            else:
                rpe_trial = rpe_current
            choices[n] = chosen_action
            rewards[n] = reward_trial
            rpe[n, :] = rpe_trial
            weight_arr[n, :] = weight_current
            bias_arr[n] = bias_current
            weight_current = self.update_w(bias_current, weight_current, choices[n], rpe_trial, params_d)
            bias_current = self.update_b(bias_current, weight_current, choices[n], rewards[n], params_d)

        data["Decision"] = choices
        data["Reward"] = rewards
        block_lengths = data.groupby(["ID", "Session", "blockNum"], as_index=False).size()
        block_lengths = block_lengths.rename(columns={"size": "blockLength"})
        data = data.merge(block_lengths, how="left", on=["ID", "Session", "blockNum"])
        data = add_switch(data)
        return data

    def emulate(self, data: pd.DataFrame, params: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """
        Emulate model behavior for the input data and parameter set.
        
        Args:
            data (pd.DataFrame): Experimental data.
            params (pd.DataFrame): Subject-specific model parameters.
        
        Returns:
            pd.DataFrame: Data with emulated decision and reward latents.
        """
        N = data.shape[0]
        q_diff, rpe, bias_arr, weight_arr = self.latents_init(N)
        choices = np.zeros(N, dtype=int)
        rewards = np.zeros(N, dtype=int)

        subjects = data["Subject"]
        subject_ids = data["ID"]
        targets = data["Target"]
        observed_choices = data["Decision"]
        observed_rewards = data["Reward"]

        condition_str = data["Condition"].unique()[0]
        p_cor, p_inc = [float(p) / 100 for p in condition_str.split("-")]
        fix_cols = ["ID", "Subject", "Session", "Trial", "blockTrial", "blockLength", "Target", "Condition"]
        emu_data = data[fix_cols].reset_index(drop=True)
        params_d = self.id_param_init(params, subject_ids.iat[0])
        bias_init, weight_init, gam = params_d["b0"], params_d["w0"], params_d["gam"]

        bias_current, weight_current = bias_init, weight_init
        marginal_prev = self.marginal_init()

        for n in range(N):
            if n > 0 and subject_ids.iat[n] != subject_ids.iat[n - 1]:
                params_d = self.id_param_init(params, subject_ids.iat[n])
            if n == 0 or subjects.iat[n] != subjects.iat[n - 1]:
                bias_current = bias_init
                weight_current = np.copy(weight_init)
                marginal_prev = self.marginal_init()
            elif data["Session"].iat[n] != data["Session"].iat[n - 1]:
                bias_current = bias_init
                weight_current = weight_init * gam + (1 - gam) * weight_current
                marginal_prev = self.marginal_init()

            qs = self.calc_q(bias_current, weight_current).ravel()
            q_diff[n] = qs[1] - qs[0]
            chosen_action = self.select_action(q_diff[n], marginal_prev, params_d)
            marginal_prev = self.update_m(chosen_action, marginal_prev, params_d)

            if chosen_action == observed_choices.iat[n]:
                reward_trial = observed_rewards.iat[n]
            else:
                roll = np.random.random()
                if targets.iat[n] == chosen_action:
                    reward_trial = int(roll <= p_cor)
                else:
                    reward_trial = int(roll <= p_inc)
            choices[n] = chosen_action
            rewards[n] = reward_trial
            rpe_current = reward_trial - qs[chosen_action]
            if self.fixed_params["CF"]:
                rpe_counterfactual = (1 - reward_trial) - qs[1 - chosen_action]
                rpe_trial = np.array([rpe_current, rpe_counterfactual])
            else:
                rpe_trial = rpe_current
            rpe[n, :] = rpe_trial
            weight_arr[n, :] = weight_current
            bias_arr[n] = bias_current
            weight_current = self.update_w(bias_current, weight_current, choices[n], rpe_trial, params_d)
            bias_current = self.update_b(bias_current, weight_current, choices[n], rewards[n], params_d)

        emu_data["Decision"] = choices
        emu_data["Reward"] = rewards
        emu_data = add_switch(emu_data)
        emu_data = self.assign_latents(emu_data, q_diff, rpe, bias_arr, weight_arr)
        return emu_data

    def fit(self, data: pd.DataFrame, *args, **kwargs):
        """
        Fit the 2ABT model parameters using maximum likelihood estimation.
        
        Returns:
            self: Fitted model object.
        
        Raises:
            RuntimeError: If the optimization fails.
        """
        data = self.fit_marginal(data)
        params_df = self.create_params(data)
        subject_ids = params_df["ID"]
        param_names = [c for c in params_df.columns if c != "ID"]

        params2x = lambda df: df.drop(columns="ID").values.ravel(order="C")
        x2params = lambda x: pd.DataFrame(
            x.reshape((-1, len(self.param_dict)), order="C"),
            columns=param_names,
            index=subject_ids
        ).reset_index().rename(columns={"index": "ID"})
        x0 = params2x(params_df)

        def neg_log_likelihood(x: np.ndarray) -> float:
            current_params = x2params(x)
            data_sim = self.sim(data.copy(), current_params, *args, **kwargs)
            valid_mask = data_sim["Decision"] != -1
            prob_sim = self.get_proba(data_sim, current_params)[valid_mask].values
            decisions = data_sim.loc[valid_mask, "Decision"].values
            epsilon = 1e-15
            prob_sim = np.clip(prob_sim, epsilon, 1 - epsilon)
            return -(decisions @ np.log(prob_sim) + (1 - decisions) @ np.log(1 - prob_sim))

        params_bounds = [self.param_dict[pn].bounds for pn in param_names] * len(subject_ids)
        method = kwargs.get("method", "L-BFGS-B")
        res = scipy.optimize.minimize(neg_log_likelihood, x0, method=method, bounds=params_bounds, tol=1e-6)
        if not res.success:
            raise RuntimeError(f"Optimization failed: {res.message}")

        self.fitted_params = x2params(res.x)
        negloglik = res.fun
        bic = len(res.x) * np.log(len(data)) + 2 * negloglik
        aic = 2 * len(res.x) + 2 * negloglik

        data_sim_opt = self.sim(data.copy(), self.fitted_params, *args, **kwargs)
        data["choice_p"] = self.get_proba(data_sim_opt)

        self.summary = {
            "bic": bic,
            "aic": aic,
            "latents": data[self.latent_names + ["choice_p"]].reset_index(drop=True),
        }
        return self