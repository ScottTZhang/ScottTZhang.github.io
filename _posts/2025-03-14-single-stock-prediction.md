1. 用stock_zh_a_hist_min_em获取（初始化时）最近三个月的分时数据（5分钟period="5"或者1小时 period="60",  且adjust="hfq"），结合stock_news_em得到最近三个月的新闻做情绪分析，结合stock_financial_debt_ths和stock_financial_benefit_ths得到的财务数据做基本面分析，对个股（SYMBOL）进行股价建模和预测。
2. 数据存储在MySQL.
```
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import akshare as ak
import pandas as pd
import pytz
from sqlalchemy import create_engine
from datetime import datetime, timedelta
from transformers import pipeline
import os

# 配置信息
SYMBOL = "******"
DB_CONFIG = {
    "host": "127.0.0.1",
    "user": "********",
    "password": "********",
    "database": "********",
    "port": ********
}

class LSTM_Model(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        assert len(x.shape) == 3, \
            f"输入必须是3D张量，得到 {len(x.shape)}D"
        h0 = torch.zeros(1, x.size(0), self.lstm.hidden_size).to(self.device)
        c0 = torch.zeros(1, x.size(0), self.lstm.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class StockDataset(Dataset):
    def __init__(self, data, seq_length=30):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_length]
        y = self.data[idx+self.seq_length, 3]  # 使用close价格作为目标
        return torch.FloatTensor(x), torch.FloatTensor([y])

class StockPipeline:
    # ... 保留原有方法 ...
    def __init__(self):
        self.engine = create_engine(
            f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="uer/roberta-base-finetuned-jd-binary-chinese")

    def _get_last_update(self, table_name):
        query = f"SELECT MAX(last_modified_date) FROM {table_name} WHERE symbol = '{SYMBOL}'"
        last_date = pd.read_sql(query, self.engine).iloc[0,0]
        return last_date or datetime(2023, 1, 1, tzinfo=pytz.utc)

    def fetch_price_data(self):
        # last_date = self._get_last_update("stock_5min").astimezone(pytz.timezone('Asia/Shanghai')) + timedelta(minutes=5)
        
         # Retrieve the last update time as a string
        last_date_str = self._get_last_update("stock_5min")
        
        # Convert the string to a datetime object
        last_date = pd.to_datetime(last_date_str)

        # Define the desired timezone
        timezone = pytz.timezone('Asia/Shanghai')

        # Check if last_date is naive or aware
        if last_date.tzinfo is None:
            # Naive datetime: localize to make it timezone-aware
            last_date = timezone.localize(last_date)
        else:
            # Aware datetime: convert to the desired timezone
            last_date = last_date.astimezone(timezone)

        # Add 5 minutes to the timestamp
        last_date += timedelta(minutes=5)

        start_date = (datetime.now(pytz.timezone('Asia/Shanghai')) - timedelta(days=180)).strftime("%Y%m%d")
        
        df = ak.stock_zh_a_hist_min_em(
            symbol=SYMBOL,
            period="5",
            adjust='hfq',
            start_date=start_date
        )
        # 处理时区转换
        df['datetime'] = pd.to_datetime(df['时间']).dt.tz_localize('Asia/Shanghai')
        df = df.rename(columns={
            '开盘': 'open',
            '最高': 'high',
            '最低': 'low',
            '收盘': 'close',
            '成交量': 'volume'
        })
        df['symbol'] = SYMBOL
        df['adjust_type'] = 'hfq'
        df['last_modified_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df = df[['symbol', 'datetime', 'open', 'high', 'low', 'close', 'volume', 'adjust_type', 'last_modified_date']]
        
        # 保存到数据库
        df = df.drop_duplicates(subset=['symbol', 'datetime'])
        df.to_sql('stock_5min', self.engine, if_exists='replace', index=False)

    def fetch_news_data(self):
        news_df = ak.stock_news_em(symbol=SYMBOL)
        news_df['pub_date'] = pd.to_datetime(news_df['发布时间']).dt.tz_localize('Asia/Shanghai')
        
        # 情感分析
        news_df['sentiment_score'] = news_df['新闻内容'].apply(
            lambda x: self.sentiment_analyzer(x[:512])[0]['score']
        )
        
        news_df = news_df.rename(columns={
            '新闻标题': 'title',
            '新闻内容': 'content'
        })
        news_df['symbol'] = SYMBOL
        news_df['last_modified_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        news_df = news_df[['symbol', 'title', 'content', 'pub_date', 'sentiment_score']]
        
        news_df = news_df.drop_duplicates(subset=['pub_date'])
        news_df.to_sql('stock_news', self.engine, if_exists='replace', index=False)
        
        
    def _convert_chinese_number(self, x):
        """
        Converts a string ending with '亿' or '万' to a numeric value.
        '亿' is multiplied by 1e8 and '万' is multiplied by 1e4.
        If x is already numeric, it returns x.
        """
        if isinstance(x, str):
            x = x.strip()
            if x.endswith("亿"):
                try:
                    return float(x.replace("亿", "")) * 1e8
                except ValueError:
                    return None
            elif x.endswith("万"):
                try:
                    return float(x.replace("万", "")) * 1e4
                except ValueError:
                    return None
            else:
                try:
                    # If it's a string but doesn't contain Chinese units,
                    # try converting directly to float.
                    return x
                except ValueError:
                    return None
        return x

    def fetch_financial_data(self):
        # 获取债务数据
        debt_df_raw = ak.stock_financial_debt_ths(symbol=SYMBOL)
        debt_df = debt_df_raw.map(self._convert_chinese_number)

        # 获取收益数据
        profit_df_raw = ak.stock_financial_benefit_ths(symbol=SYMBOL)
        profit_df = profit_df_raw.map(self._convert_chinese_number)
        
        # 合并财务数据
        merged_df = pd.merge(debt_df, profit_df, on='报告期')
        merged_df['report_date'] = pd.to_datetime(merged_df['报告期'] + ' ' + '15:00:00')
        merged_df['symbol'] = SYMBOL
        merged_df['last_modified_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # '资产负债率' 不存在. 计算资产负债率，并添加为新列
        merged_df['*负债合计'] = pd.to_numeric(merged_df['*负债合计'], errors='coerce')
        merged_df['*资产合计'] = pd.to_numeric(merged_df['*资产合计'], errors='coerce')
        merged_df['debt_ratio'] = merged_df['*负债合计'] / merged_df['*资产合计']
        
        merged_df = merged_df[['symbol', 'report_date', 'debt_ratio', '*净利润', '（一）基本每股收益']]
        
        # 'symbol', 'report_date', '资产负债率', '净利润', '每股收益'
        merged_df = merged_df.rename(columns={'*净利润': 'net_profit', '（一）基本每股收益': 'eps'})
        
        merged_df.to_sql('financial_data', self.engine, if_exists='replace', index=False)
        
    def _prepare_training_data(self):
        # 获取合并后的特征数据
        query = f"""
        SELECT s.datetime, s.open, s.high, s.low, s.close, s.volume, 
           COALESCE(n.sentiment_score, 0.5) as sentiment_score,
           COALESCE(f.debt_ratio, (SELECT AVG(debt_ratio) FROM financial_data)) as debt_ratio
        FROM stock_5min s
        LEFT JOIN stock_news n 
            ON s.symbol = n.symbol 
            AND n.pub_date BETWEEN s.datetime - INTERVAL 4 HOUR AND s.datetime
        LEFT JOIN financial_data f 
            ON s.symbol = f.symbol 
            AND YEAR(s.datetime) = YEAR(f.report_date)
            AND QUARTER(s.datetime) = QUARTER(f.report_date)
        WHERE s.symbol = '{SYMBOL}'
        ORDER BY s.datetime
        """
        raw_data = pd.read_sql(query, self.engine)
        
        # 处理空值
        raw_data['sentiment_score'].fillna(0.5, inplace=True)  # 中性情绪
        raw_data['debt_ratio'].fillna(method='ffill', inplace=True)
        
        # 在数据预处理阶段添加异常值过滤
        raw_data = raw_data[(raw_data['volume'] > 0) & 
            (raw_data['close'] > 0) &
            (raw_data['debt_ratio'].between(0, 1))]
        
        raw_data['price_momentum'] = raw_data['close'].pct_change(periods=48)  # 1日动量
        
        # 特征工程
        features = raw_data[['open', 'high', 'low', 'close', 'volume', 
                            'sentiment_score', 'debt_ratio', 'price_momentum']]
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = self.scaler.fit_transform(features)
        
        return [raw_data, scaled_data]

    def _generate_future_dates(self, last_date):
        current_date = last_date
        while True:  # 持续生成日期
            current_date += timedelta(minutes=5)
            # 仅包含交易日（周一至周五）
            if current_date.weekday() < 5:
                yield current_date
                
    def _get_historical_volume_profile(self):
        # 获取历史各时间段的平均成交量"""
        query = f"""
            SELECT 
                DATE_FORMAT(datetime, '%%H%%i') AS time_key,
                AVG(volume) AS avg_volume
            FROM stock_5min
            WHERE symbol = %s
            GROUP BY time_key
        """
        df = pd.read_sql(query, self.engine, params=(SYMBOL,))
        return df.set_index('time_key')['avg_volume'].to_dict()
        
    # 新增辅助方法: 模拟每5分钟的成交量
    def _calculate_volume(self, last_volume, current_price, last_price, current_time, base_volume):
        # 时间周期因子（改为增强流动性）
        hour = current_time.hour
        if 9 <= hour < 11 or 13 <= hour < 15:
            time_factor = 1.05  # 交易时段增加5%流动性
        else:
            time_factor = 0.98  # 非交易时段仅微降2%

        # 价格变动因子（增强响应）
        price_change = (current_price - last_price) / last_price
        change_factor = 1.0 + 2.0 * np.tanh(price_change / 0.01)  # ±1%价格变动对应±200%成交量变化
        
        # 历史基准因子（均值回归）
        history_factor = np.sqrt(base_volume / (last_volume + 1e-5))  # 平方根平滑
        
        # 综合计算
        new_volume = last_volume * time_factor * np.clip(change_factor, 0.5, 3.0) * history_factor
        
        # 波动增强（±20%随机性）
        new_volume *= np.random.uniform(0.8, 1.2)
        
        return np.clip(new_volume, 100, 2*base_volume)  # 限制在历史基准2倍内
    
    def generate_report(self):
        from sqlalchemy import text
        query = text("TRUNCATE TABLE price_forecast")
        with self.engine.begin() as connection:
            connection.execute(query)
        # 准备数据
        [raw_data, scaled_data] = self._prepare_training_data()
        
        print("缺失值统计：")
        print(raw_data.isnull().sum())

        print("数据分布：")
        print(raw_data.describe())

        print("标准化后数据范围：")
        print("Min:", scaled_data.min(axis=0))
        print("Max:", scaled_data.max(axis=0))
        
        # 训练模型
        dataset = StockDataset(scaled_data)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True, pin_memory=True)
        
        model = LSTM_Model(input_size=scaled_data.shape[1])
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
        
        # GPU加速和混合精度配置
        scaler = torch.amp.GradScaler('cuda',enabled=torch.cuda.is_available())
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        
        # 改进后的训练循环
        best_loss = float('inf')
        for epoch in range(200):
            model.train()
            total_loss = 0
            gradient_norms = []
            
            # 使用tqdm进度条
            from tqdm import tqdm
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/200")
            
            for inputs, labels in pbar:
                inputs = inputs.to(model.device, non_blocking=True)
                labels = labels.to(model.device, non_blocking=True)
                
                # 混合精度前向传播
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                # 反向传播和梯度裁剪
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                
                # 梯度值裁剪（核心修改）
                torch.nn.utils.clip_grad_value_(model.parameters(), 5.0) # 从1.0放宽到5.0
                
                # 梯度范数监控
                total_norm = torch.norm(
                    torch.stack([torch.norm(p.grad.detach()) for p in model.parameters() if p.grad is not None])
                )
                gradient_norms.append(total_norm.item())
                
                # 参数更新
                scaler.step(optimizer)
                scaler.update()
                
                # 损失统计
                total_loss += loss.item()
                pbar.set_postfix({
                    "Loss": f"{total_loss/(pbar.n+1):.4f}",
                    "Grad Norm": f"{total_norm.item():.4f}"
                })
            
            # 早停机制
            avg_loss = total_loss / len(dataloader)
            if avg_loss < best_loss:
                best_loss = avg_loss
            else:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
        # 预测未来数据
        last_sequence = scaled_data[-30*6:]
        predictions = []
        current_seq = torch.FloatTensor(last_sequence).unsqueeze(0).to(model.device)
        
        # 生成预测时间序列
        predict_steps = 5 * 48  # 5天×48个5分钟段
        date_generator = self._generate_future_dates(
            pd.to_datetime(raw_data['datetime'].iloc[-1]).to_pydatetime()
        )
        
        # 改进的预测循环
        reconstructed_data = np.zeros((predict_steps, scaled_data.shape[1]))
        last_valid_close = last_sequence[-1, 3]
        
        # 获取历史成交量模式（新增方法）
        historical_volume_profile = self._get_historical_volume_profile()
    
        # 生成时间序列
        future_dates = []  # 只取5个交易日
        
        for step in range(predict_steps):
            with torch.no_grad():
                pred = model(current_seq).cpu().numpy()[0][0] + np.random.normal(0, 1e-6)
        
            # 生成合理的新特征
            last_features = current_seq.cpu().numpy()[0, -1]  # 取最后一个时间步的特征
            
            # 3.13: 生成当前预测时间（重要修改）
            current_date = next(date_generator)  # 提前获取当前预测时间
            future_dates.append(current_date)  # 存储到日期列表
        
            # 3.13: 获取历史同时间段成交量基准（新增）
            time_key = f"{current_date.hour:02d}{current_date.minute:02d}"
            base_volume = historical_volume_profile.get(time_key, last_features[4])
            
            # 3.13: 动态成交量计算（核心改进）
            new_volume = self._calculate_volume(
                last_volume=last_features[4],
                current_price=pred,
                last_price=last_features[3],
                current_time=current_date,
                base_volume=base_volume
            )
            print ("new volume:", new_volume)
            new_open = last_features[3] * np.random.uniform(0.995, 1.005)  # 允许±0.5%跳空
            new_high = max(
                new_open * 1.005,  # 最高可达开盘价+0.5%
                pred * 1.002       # 或预测价+0.2%
            )
            new_low = min(
                new_open * 0.995,  # 最低至开盘价-0.5%
                pred * 0.998       # 或预测价-0.2%
            )
            
            # 3.14: 新增特征：价格动量（示例）
            price_momentum = (pred - last_features[3]) / last_features[3]  # 当前价格变动率
    
            new_features = np.array([
                new_open,
                new_high,
                new_low,
                pred,
                new_volume,
                last_features[5],
                last_features[6],
                price_momentum
            ])
            
            # 修正维度问题（关键修改点）
            new_tensor = torch.FloatTensor(new_features).unsqueeze(0).unsqueeze(0).to(model.device)
        
            # 正确的拼接操作
            current_seq = torch.cat([
                current_seq[:, 1:, :],  # 保持三维结构 [1, seq_len-1, features]
                new_tensor               # 新特征 [1, 1, features]
            ], dim=1)  # 在序列长度维度拼接
                
            # 存储重建数据
            reconstructed_data[step] = new_features
            predictions.append(pred)
            last_valid_close = pred
        
        # 逆标准化处理
        predicted_prices = self.scaler.inverse_transform(reconstructed_data)[:, 3]
    
        # 打印预测结果
        print("\n预测结果（未来5个时间点股价）：")
        forecast_list = []
        for date, price in zip(future_dates, predicted_prices[:5]):
            print(f"{date.strftime('%Y-%m-%d %H:%M')}: {price:.2f}")
            forecast_list.append({
                "date": date.strftime('%Y-%m-%d %H:%M'),
                "price": round(float(price), 2)
            })
        
        # 保存预测结果
        forecast_df = pd.DataFrame({
            'symbol': SYMBOL,
            'forecast_date': [d.strftime('%Y-%m-%d %H:%M:%S') for d in future_dates],
            'predicted_close': predicted_prices,
            'forecast_time': datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S'),
            'last_modified_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        forecast_df.to_sql('price_forecast', self.engine, if_exists='replace', index=False)
        
        # 3.13: 在generate_report末尾添加成交量分析
        volume_df = pd.DataFrame({
            'datetime': future_dates,
            'predicted_volume': reconstructed_data[:, 4],
            'historical_avg': [historical_volume_profile.get(d.strftime('%H%M'), np.nan) for d in future_dates]
        })

        # print(reconstructed_data)
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        # 指定支持中文的字体，例如 SimHei
        plt.rcParams['font.sans-serif'] = ['SimHei']
        # 修正负号显示问题
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(15, 5))
        plt.plot(volume_df['datetime'], volume_df['predicted_volume'], label='预测成交量')
        plt.plot(volume_df['datetime'], volume_df['historical_avg'], 'g--', label='历史平均')
        plt.title('成交量预测对比')
        plt.legend()
        plt.savefig('volume_comparison.png')
        plt.close()
        # 生成可视化图表
        self._generate_visualization(raw_data, predicted_prices, future_dates)
            
        return forecast_list  # 返回预测结果列表
        
    def _generate_visualization(self, historical, predictions, dates):
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        plt.figure(figsize=(18, 8))
        
        # 历史数据（最后3天）
        historical = historical.iloc[-3*48*3:]  # 保留最后3天数据（每天约48*3=144个5分钟段）
        plt.plot(historical['datetime'], historical['close'], label='Historical Price', alpha=0.7)

        # 预测数据（全部240个点）
        plt.plot(dates, predictions, 'r-', linewidth=1.5, label='Predicted Price')

        # 标注关键时间点
        important_dates = [dates[0], dates[-1]]  # 首尾时间点
        for date in important_dates:
            idx = dates.index(date)
            plt.annotate(f'{predictions[idx]:.2f}',
                         xy=(date, predictions[idx]),
                         xytext=(0, 10),
                         textcoords='offset points',
                         ha='center',
                         color='red',
                         fontsize=20)

        # 优化时间轴显示
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        
        plt.title(f'{SYMBOL} 5-Day Price Forecast (5-Minute Intervals)')
        plt.xlabel('Date/Time (Asia/Shanghai)')
        plt.ylabel('Price')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        # 保存图片
        reports_dir = os.path.join(os.getcwd(), 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        save_path = os.path.join(reports_dir, f'{datetime.now().strftime("%Y%m%d%H%M%S")}_forecast.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    def run_pipeline(self):
        self.fetch_price_data()
        self.fetch_news_data()
        self.fetch_financial_data()
        self.generate_report()
        
if __name__ == "__main__":
    pipeline = StockPipeline()
    pipeline.run_pipeline()
```
