import pandas as pd
import numpy as np
import copy
from pyspark.sql import SparkSession

spark = SparkSession.builder.master('yarn').appName('test').config('spark.executor.memory','10gb').getOrCreate()
zq_model2_data_0805_label_1 = spark.sql("select * from jtstyf_rundb.zq_model2_data_0805_label_1")
zq_model2_data_0805_label_0 = spark.sql("select * from jtstyf_rundb.zq_model2_data_0805_label_0")

df0 = zq_model2_data_0805_label_0.toPandas()
df1 = zq_model2_data_0805_label_1.toPandas()
data = pd.concat([df0, df1], ignore_index=True)

data['label'].value_counts()

data.rename(columns={
    'IS_VIP_USER':'重要客户标识',
    'VIP_CODE':'会员级别',
    'AGE':'年龄',
    'SEX':'性别',
    'profession_code':'职业编码',
    'education_code':'教育程度编码',
    'marital_status':'婚姻状况',
    'cust_star_lvl':'客户星级',
    'brand_id':'品牌标识',
    'is_m2m_yx':'是否物联网标识（亚信口径）',
    'is_m2m_td':'是否物联网标识（TD口径）',
    'user_star_lvl':'用户星级分值',
    'is_village_user':'是否农村用户',
    'is_school_user':'是否校园用户',
    'is_group':'是否集团用户',
    'raise_card_id':'疑似养卡标识',
    'user_status':'用户状态',
    'innet_months':'在网时长（月数）',
    'stop_days':'停机时长(天)',
    'mon_halt_cnt':'月停机次数',
    'year_total_halt_cnt':'年累计停机次数',
    'msisdn_m6_halt_rate':'手机号码近6个月总停机频次',
    'msisdn_owe_halt_rate':'手机号码欠费停机频次',
    'user_m6_owe_halt_cnt':'用户近6个月欠费停机次数',
    'mon_break_id':'月流失标识',
    'package_sms':'套内短信条数',
    'package_voice':'套内语音分钟数',
    'package_flux':'套内流量数',
    'term_brand':'终端品牌',
    'this_acct_fee_tax':'当月消费总金额',
    'this_flux_days':'月上网天数',
    'gprs_total_flux':'月总使用流量',
    'flux_sj_roam_days':'当月省内漫游流量天数',
    'flux_gj_roam_days':'当月省际漫游流量天数',
    'sms_cnt':'短信条数',
    'sms_user_cnt':'短信人数',
    'sms_circle_user_cnt':'短信交往圈联系人数',
    'OUT_IN_SMS_CIRCLE_USER_CNT':'短信交往圈双向重合人数',
    'AVG_SMS_CNT':'交往圈人均短信条数',
    'avg_ptp_sms_cnt':'交往圈人均点对点短信条数',
    'out_sms_days':'短信发送天数',
    'ptp_sms_days':'点对点短信发送天数',
    'sms_rate':'短信频度',
    'BEL_SAME_OPPO_CNT':'当月呼出出对端和本机归属地相同的不重复对端数',
    'RESID_SAME_OPPO_CNT':'当月呼出出对端和本机常驻地相同的不重复对端数',
    'MAX_DAY_IN_VOICE_CNT':'当月内用户日最高被叫通话次数',
    'MAX_DAY_OUT_VOICE_CNT':'当月内用户日最高主叫通话次数',
    'MAX_DAY_IN_VOICE_DURA':'当月内用户日最高被叫通话时长',
    'MAX_DAY_OUT_VOICE_DURA':'当月内用户日最高主叫通话时长',
    'PTP_SMS_CNT':'当月点对点短信下行量',
    'MAX_DAY_PTP_SMS_CNT':'当月内日最高点对点短信下行量',
    'RESID_CITY_DAY_STAY_DAYS':'当月白天常驻城市停留天数',
    'RESID_CITY_NIGHT_STAY_DAYS':'当月夜间常驻城市停留天数',
    'RESID_DAY_CITY_CNT':'当月白天常驻过城市数里（剔重）',
    'RESID_NIGHT_CITY_CNT':'当月夜间常驻过城市数里（剔重）',
    'OPPO_BEL_CITY_CNT':'当月与该号码通话的对端所属城市数量',
    'OPPO_MAX_CITY_NAME':'当月与该号码通话次数最多的城市名称',
    'COMM_BASESTA_CNT':'当月通信基站个数',
    'ACT_CELL_JF_DURA':'主要活动小区呼出计费时长',
    'ACT_CELL_VOICE_CNT':'主要活动小区呼出通话次数',
    'ACT_CELL_FLUX':'主要活动小区使用流量',
    'FRE_TERM_MSISDN_CNT':'当月用户最常用终端附着号码量',
    'TERM_MSISDN_CNT':'当月用户终端附着号码量',
    'VOICE_DURA':'通话时长',
    'OUT_VOICE_DURA':'主叫通话时长',
    'IN_VOICE_DURA':'被叫通话时长',
    'VOICE_CNT':'通话次数',
    'OUT_VOICE_CNT':'主叫通话次数',
    'IN_VOICE_CNT':'被叫通话次数',
    'TRANS_OUT_VOICE_CNT':'呼转通话次数',
    'VOICE_CIRCLE_USER_CNT':'语音交际圈人数',
    'OUT_USER_CNT':'主叫人数',
    'IN_USER_CNT':'被叫人数',
    'PRESENT_PKG_USED_VOICE':'套餐外语音使用量',
    'VOICE_DAYS':'通话天数',
    'VOICE_BASESTA_CNT':'语音月基站个数'
    },inplace=True)

data['label'] = data['label'].astype('int')
data.drop(['msisdn','年累计停机次数','会员级别','是否物联网标识（亚信口径）','是否物联网标识（TD口径）','终端品牌','当月与该号码通话次数最多的城市名称','品牌标识'], axis=1, inplace=True)
data.replace(r'\N',np.NAN,inplace=True)
data.replace(r'\n',np.NAN,inplace=True)
data.replace('None',np.NAN,inplace=True)
data.replace('nan',np.NAN,inplace=True)
data.replace('NaN',np.NAN,inplace=True)
data.replace('',np.NAN,inplace=True)
(data=='None').any().any()
data.isnull().any().any()

str_list = ['重要客户标识','性别','职业编码','教育程度编码','婚姻状况','客户星级','品牌标识','是否农村用户','是否校园用户','是否集团用户','疑似养卡标识','用户状态','月流失标识','短信频度']
for col in list(data.columns)[:-1]:
    if col in str_list:
        data[col] = data[col].astype('str')
    else:
        data[col] = data[col].astype('float')

data.replace(r'\N',np.NAN,inplace=True)
data.replace(r'\n',np.NAN,inplace=True)
data.replace('None',np.NAN,inplace=True)
data.replace('nan',np.NAN,inplace=True)
data.replace('NaN',np.NAN,inplace=True)
data.replace('',np.NAN,inplace=True)
(data=='None').any().any()
data.isnull().any().any()

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

missing_rate = data.isnull().sum() / len(data)
droped_columns = missing_rate[missing_rate > 0.2].index.tolist()
logger.info(f"剔除缺失率大于0.2的特征数量: {len(droped_columns)}")
data.drop(droped_columns, inplace=True, axis=1)
logger.info(f"剔除缺失率大于0.2的特征：{droped_columns}")

zero_rate = ((data==0) | (data=='0')).sum() / len(data)
zero_droped_columns = zero_rate[zero_rate > 0.3].index.tolist()
logger.info(f"剔除0值占比大于0.3的特征数量: {len(zero_droped_columns)}")
logger.info(f"剔除的0值特征：{zero_droped_columns}")
data.drop(zero_droped_columns[:-1], inplace=True, axis=1)
logger.info(f"剔除0值占比大于0.3的特征：{zero_droped_columns}")

bad_missing_ratio = data[data['label']==1].isnull().sum(axis=1)/data.shape[1]
bad_data_index = bad_missing_ratio[bad_missing_ratio == 0].index.tolist()
logger.info(f"坏样本数据索引数量: {len(bad_data_index)}")
good_missing_ratio = data[data['label']==0].isnull().sum(axis=1)/data.shape[1]
good_data_index = good_missing_ratio[good_missing_ratio == 0].index.tolist()
logger.info(f"好样本数据索引数量: {len(good_data_index)}")


data_bad = data.iloc[bad_data_index]
data_good = data.iloc[good_data_index]
data_all = pd.concat([data_bad,data_good],ignore_index=True)
data_all

data_good_ratio_1_1 = data.iloc[good_data_index].sample(n=len(data_bad),random_state=42)
data_ratio_1_1 = pd.concat([data_bad,data_good_ratio_1_1],ignore_index=True)
data_ratio_1_1

data_good_ratio_1_2 = data.iloc[good_data_index].sample(n=len(data_bad)*2,random_state=42)
data_ratio_1_2 = pd.concat([data_bad,data_good_ratio_1_2],ignore_index=True)
data_ratio_1_2

data_good_ratio_1_3 = data.iloc[good_data_index].sample(n=len(data_bad)*3,random_state=42)
data_ratio_1_3 = pd.concat([data_bad,data_good_ratio_1_3],ignore_index=True)
data_ratio_1_3

data_good_ratio_1_4 = data.iloc[good_data_index].sample(n=len(data_bad)*4,random_state=42)
data_ratio_1_4 = pd.concat([data_bad,data_good_ratio_1_4],ignore_index=True)
data_ratio_1_4

data_good_ratio_1_5 = data.iloc[good_data_index].sample(n=len(data_bad)*5,random_state=42)
data_ratio_1_5 = pd.concat([data_bad,data_good_ratio_1_5],ignore_index=True)
data_ratio_1_5

data_all.to_csv('./zq_model/fraud_model/data/data_all.csv', index=False)
data_ratio_1_1.to_csv('./zq_model/fraud_model/data/data_ratio_1_1.csv', index=False)
data_ratio_1_2.to_csv('./zq_model/fraud_model/data/data_ratio_1_2.csv', index=False)
data_ratio_1_3.to_csv('./zq_model/fraud_model/data/data_ratio_1_3.csv', index=False)
data_ratio_1_4.to_csv('./zq_model/fraud_model/data/data_ratio_1_4.csv', index=False)
data_ratio_1_5.to_csv('./zq_model/fraud_model/data/data_ratio_1_5.csv', index=False)

