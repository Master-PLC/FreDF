from .exp_long_term_forecasting import Exp_Long_Term_Forecast
from .exp_imputation import Exp_Imputation
from .exp_short_term_forecasting import Exp_Short_Term_Forecast

EXP_DICT = {
    'long_term_forecast': Exp_Long_Term_Forecast,
    'short_term_forecast': Exp_Short_Term_Forecast,
    'imputation': Exp_Imputation
}