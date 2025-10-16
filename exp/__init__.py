from .exp_long_term_forecasting import Exp_Long_Term_Forecast
from .exp_long_term_forecasting_meta_ml3 import Exp_Long_Term_Forecast_META_ML3
from .exp_imputation import Exp_Imputation
from .exp_short_term_forecasting import Exp_Short_Term_Forecast

EXP_DICT = {
    'long_term_forecast': Exp_Long_Term_Forecast,
    'long_term_forecast_meta_ml3': Exp_Long_Term_Forecast_META_ML3,
    'short_term_forecast': Exp_Short_Term_Forecast,
    'imputation': Exp_Imputation
}