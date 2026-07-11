import numpy as np
from sklearn.base import BaseEstimator
from tqdm import tqdm
from afisp.utils import cohens_d, bootstrap_ci
from statsmodels.stats.weightstats import ttest_ind
from imodels.rule_set.skope_rules import SkopeRulesClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
from sirus import SirusClassifier, sirus_cv
import pandas as pd


class SubgroupPhenotyper(BaseEstimator):
    """This class performs subgroup phenotyping. After using stability analysis
    to identify a poorly performing data subset, the SubgroupPhenotyper is used
    to find specific data phenotypes or subgroups that are present within the
    data subset.
    """

    def __init__(self):
        """Constructor method
        """
        self.fit_called_ = False

    def fit(self, subgroup_feature_data, subset_labels, test_loss,
            method="SIRUS", depth=2, cv=False, rule_max=50,
            p0=0.025, random_state=None, verbose=0):
        """Computes the subgroup phenotypes using an interpretable classifier.
        The subgroup phenotyper expects categorical features to be encoded as
        binary dummy variables.

        :param subgroup_feature_data: DataFrame containing the subgroup feature
            data. All columns must be numeric (encode categorical features as
            binary dummy variables).
        :param subset_labels: Binary labels for whether each sample is in the
            worst data subset.
        :param test_loss: The observed loss for each sample. Used for filtering
            rules based on statistical significance and effect size.
        :param method: Selects the interpretable classification method used for
            extracting the subgroup phenotypes. "SIRUS" uses the 'Stable and
            Interpretable RUle Set' method (the pure-Python ``sirus`` package).
            This is recommended and is the default. As an alternative, the
            "DecisionList" method uses the imodels SkopeRulesClassifier. Defaults
            to "SIRUS".
        :param cv: For method SIRUS, whether or not to use cross-validation to
            select the rule selection threshold. If True, then p0 is ignored.
            Using cross-validation can be very slow, but results are often
            better.
        :param p0: For method SIRUS, the threshold (between 0 and 1) for rule
            selection. Ignored if cv is True, since cv is used to determine a
            good value for p0. The higher the value of p0, the fewer rules that
            will be selected. Recommended to choose a value < 0.1. Pass None (or
            a non-positive value) to let SIRUS choose the number of rules
            automatically. Defaults to 0.025.
        :type p0: Float in (0, 1)
        :param rule_max: The max number of rules that SIRUS will consider,
            defaults to 50.
        :type rule_max: int > 0
        :param depth: The maximum number of literals (splits) per rule, defaults
            to 2.
        :param random_state: Seed for reproducible SIRUS fits. Note that the
            Python ``sirus`` package is an independent re-implementation of the
            R package; a fixed seed makes runs reproducible but does not
            reproduce the R package's exact rules.
        :return: The candidate rules extracted by the Subgroup Phenotyper
        :rtype: List[string]
        """
        phenotype_df = subgroup_feature_data.copy()
        phenotype_df['subset_label'] = subset_labels
        self._phenotype_df = phenotype_df

        if method == "SIRUS":
            X = subgroup_feature_data
            y = np.asarray(subset_labels).astype(int)
            self._validate_sirus_inputs(subgroup_feature_data, y, cv, p0)
            if verbose > 0:
                print("Beginning call to SIRUS. If cv == True this may take a long time.")
            if cv:
                res = sirus_cv(X, y, task="classification",
                               num_rules_max=rule_max, max_depth=depth,
                               random_state=random_state, verbose=(verbose > 0))
                model = SirusClassifier(p0=res.p0_pred, max_depth=depth,
                                        num_rules_max=rule_max,
                                        random_state=random_state)
            elif p0 is None or p0 <= 0:
                # sentinel: let SIRUS choose the number of rules
                model = SirusClassifier(max_depth=depth, num_rules_max=rule_max,
                                        random_state=random_state)
            else:
                # Parity with the original R script: its explicit-p0 branch did
                # not pass num.rule.max, so the historical default (cv=False,
                # p0=0.025) capped at SIRUS's default (25). Omit num_rules_max
                # here to preserve that behavior.
                model = SirusClassifier(p0=p0, max_depth=depth,
                                        random_state=random_state)
            model.fit(X, y)
            if verbose > 0:
                print("Finished call to SIRUS")
            candidate_rules = self._extract_sirus_rules(model)
        elif method == "DecisionList":
            # Add arguments for SkopesRulesClassifier
            if depth == 1:
                md = 1
            else:
                md = list(range(1, depth+1))
            sp = SkopeRulesClassifier(max_depth=md)
            sp.fit(subgroup_feature_data.values, 
                   subset_labels, 
                   feature_names=subgroup_feature_data.columns)
            candidate_rules = sp.rules_

        else:
            raise RuntimeError('Method not implemented. Please choose one of "SIRUS" or "DecisionList"')

        if verbose > 0:
            print("Computing p-values")
        rule_p_values = self._precompute_p_values(candidate_rules, phenotype_df, test_loss)
        significant_rules = self._holm_bonferroni_correction(rule_p_values)
        if verbose > 0:
            print("Effect size filtering")
        extracted_rules = self._effect_size_filtering(significant_rules, phenotype_df, test_loss, 
                                                        effect_threshold=0.3)
        self.fit_called_ = True
        self._extracted_rules = extracted_rules
        return self._extracted_rules

    def generate_subgroup_table(self, y_test, test_preds, loss_fn=brier_score_loss):
        """Generates a summary table reporting the subgroup phenotypes
        identified to have poor performance by the SubgroupPhenotyper. The
        table also reports the sample size and the performance (including a 95%
        bootstrap confidence interval)

        :param y_test: The true labels for the test dataset
        :param test_preds: Test set predictions from model being evaluated.
        :param loss_fn: Loss function for computing average performance on test
            dataset.
        """

        if not self.fit_called_:
            raise RuntimeError('Must call "fit" on SubgroupPhenotyper object first.') 

        r_aucs = []
        r_ls = []
        r_us = []
        ns = []
        
        for rule in self._extracted_rules:
            rows = self._phenotype_df.eval(str(rule))
            ns.append(np.sum(rows))
            m, l, u = bootstrap_ci(y_test[rows], test_preds[rows], loss=loss_fn)
            r_aucs.append(m)
            r_ls.append(l)
            r_us.append(u)

        return pd.DataFrame({'Phenotype': self._extracted_rules, 
                             'Performance': r_aucs, 
                             'N': ns, 
                             'Lower': r_ls, 
                             'Upper': r_us}).sort_values(by='Performance')
        
        

    def _negate_simple_rule(self, rule):
        """Negates a rule string of the form "if ARG [<|>|<=|>=] VAL"

        Args:
            rule: A string of a rule of the form "if ARG [<|>|<=|>=] VAL"
        Returns:
            String in which the condition [>|<|>=|<=] has been negated
        """
        if ">=" in rule:
            return(rule.replace(">=", "<"))
        if "<=" in rule:
            return(rule.replace("<=", ">"))
        if ">" in rule:
            return(rule.replace(">", "<="))
        if "<" in rule:
            return(rule.replace("<", ">="))

    def _validate_sirus_inputs(self, subgroup_feature_data, y, cv, p0):
        """Validates the inputs to the SIRUS phenotyping method."""
        if not isinstance(subgroup_feature_data, pd.DataFrame):
            raise TypeError("subgroup_feature_data must be a pandas DataFrame "
                            "for method='SIRUS'.")
        non_numeric = [c for c in subgroup_feature_data.columns
                       if not pd.api.types.is_numeric_dtype(subgroup_feature_data[c])]
        if non_numeric:
            raise ValueError(
                "All subgroup_feature_data columns must be numeric for "
                f"method='SIRUS'; found non-numeric columns {non_numeric}. "
                "Encode categorical features as binary dummy variables.")
        n_classes = np.unique(y).size
        if n_classes != 2:
            raise ValueError(
                "subset_labels must contain exactly two classes for "
                f"method='SIRUS'; found {n_classes}.")
        if not cv and p0 is not None and p0 >= 1:
            raise ValueError(f"p0 must be in (0, 1); got {p0}.")

    def _extract_sirus_rules(self, model):
        """Converts the rules of a fitted SIRUS model into pandas-eval rule
        strings, orienting each rule toward the worst subset (mirrors the
        orientation/negation logic of the former R text parser).
        """
        rules = {}  # ordered dedup (model.rules_ is frequency-sorted)
        for r in model.rules_:
            for c in r.conditions:
                if c.op == "in":
                    # categorical -> not a valid pandas-eval expression
                    raise ValueError(
                        "SIRUS produced a categorical condition; encode "
                        "categorical features as numeric dummy variables "
                        "before phenotyping.")
            rule_str = " & ".join(str(c) for c in r.conditions)
            if r.output_in > r.output_out:  # rule already points at the worst subset
                rules[rule_str] = None
            elif len(r.conditions) > 1:  # negate a compound rule: ~(X & Y) = ~X or ~Y
                for c in r.conditions:
                    rules[self._negate_simple_rule(str(c))] = None
            else:  # negate a simple rule
                rules[self._negate_simple_rule(rule_str)] = None
        return list(rules)
    
    def _precompute_p_values(self, sirus_rules, phenotype_df, test_loss, alpha=0.05):
        # precompute p_values
        rule_p_values = []
        for rule in tqdm(sirus_rules):
            rows = phenotype_df.eval(str(rule))
                
                
            # two independent sample t test that brier score is larger in group than out of group
            pval = ttest_ind(test_loss[rows], 
                            x2=test_loss[~rows], 
                            value=0.,
                            alternative='larger',
                            usevar='unequal')[1]
            # check for nan
            
            # print(rule, rows.sum())
            # print(f"Mean AUC {np.mean(bootstrap_aucs):.3f} Threshold AUC {performance_threshold:.3f} p value {pval:.2f} ")
            rule_p_values.append((rule, pval, rows.sum()))
        # rule_p_values = sorted(rule_p_values, key=lambda x: x[1])
        # filter out nans
        rule_p_values = sorted([x for x in rule_p_values if not np.isnan(x[1])], key=lambda x: x[1])
        return(rule_p_values)
    
    def _holm_bonferroni_correction(self, rule_p_values, sig_value=0.05):
        # expect p-values to be sorted, smallest to largest
        
        m = len(rule_p_values)
        
        significant_rules = []
        
        for k in range(1, m+1):
            rule_k = rule_p_values[k-1]
            # reject with adaptive significance level
            if rule_k[1] < sig_value / (m + 1 - k):
                significant_rules.append(rule_k)
                continue
            # print(k, rule_k, sig_value / (m + 1 - k))
            break

        return(significant_rules)
    
    def _effect_size_filtering(self, significant_rules, phenotype_df, 
                               test_loss, effect_threshold = 0.4, 
                               verbose=False):
        rules = []
        for rule in tqdm(significant_rules):
            r = rule[0]
            rows = phenotype_df.eval(str(r))
            cd = cohens_d(test_loss[rows], test_loss[~rows])
            if verbose:
                print(f"{r} Cohen's d {cd:.2f}")
            if cd > effect_threshold:
                rules.append((r, cd))
        # sort from highest effect size to lowest
        rules = sorted(rules, key=lambda x: x[1], reverse=True)
        rules = [r[0] for r in rules] # drop the effect sizes
        return(rules)


