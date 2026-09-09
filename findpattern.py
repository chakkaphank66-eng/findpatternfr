import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta

# ==========================================
# 1. การตั้งค่าหน้าตา UI (Dual-Timeframe Mode)
# ==========================================
st.set_page_config(page_title="Gold Wick-Sweep Optimizer (H1+M5)", layout="wide")

st.sidebar.header("🕰️ Time Travel & Lab Optimizer")
st.sidebar.markdown("**Grid Search หาระยะเข้าเทรดที่ดีที่สุด**")

# อัปโหลด H1 (บังคับ)
uploaded_h1 = st.sidebar.file_uploader("📂 1. อัปโหลดไฟล์ H1 (หลัก)", type=["csv"], accept_multiple_files=True)
# อัปโหลด M5 (ทางเลือก)
uploaded_m5 = st.sidebar.file_uploader("📂 2. อัปโหลดไฟล์ M5 (เพิ่มความแม่นยำ - Optional)", type=["csv"], accept_multiple_files=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**📅 กำแพงเวลา (Time Wall Constraints)**")
sim_start_date = st.sidebar.date_input("Simulation Start Date (วันตั้งกำแพงเวลา)")
lookback_days = st.sidebar.number_input("Lookback Training (อดีตสำหรับสแกนหาค่า/วัน)", min_value=30, value=90, step=30)
forward_days = st.sidebar.number_input("Forward Testing (อนาคตสำหรับพิสูจน์/วัน)", min_value=10, value=30, step=10)

st.sidebar.markdown("---")
st.sidebar.markdown("**⚙️ ด่านอรหันต์คัดกรองกลยุทธ์ (Strict Constraints)**")
min_trades_per_week = st.sidebar.number_input("ความถี่ขั้นต่ำ (ไม้/สัปดาห์)", value=1.5, step=0.5)
target_max_dd = st.sidebar.number_input("Max Drawdown ยอมรับได้สูงสุด (R)", value=6.0, step=0.5)

run_button = st.sidebar.button("🚀 Run Full Optimizer")

# ==========================================
# 2. ฟังก์ชันหลัก (Dual-Engine Backtest)
# ==========================================
def optimize_wick_sweep(df_h1, df_m5, start_date, end_date, weeks_in_period):
    """
    แกนกลางจำลองเทรด รองรับทั้ง H1 ล้วน และ H1+M5
    """
    mask_h1 = (df_h1['time'] >= start_date) & (df_h1['time'] < end_date)
    w_h1 = df_h1[mask_h1].reset_index(drop=True)
    
    if len(w_h1) < 50:
        return []

    # ข้อมูล H1 สำหรับ Setup
    h1_times = w_h1['time'].values
    h1_hours = w_h1['time'].dt.hour.values
    avg_wicks = w_h1['Avg_Wick'].values
    h1_closes = w_h1['Close'].values
    
    # ถ้ามี M5 ให้เตรียมข้อมูล M5
    use_m5 = df_m5 is not None and not df_m5.empty
    if use_m5:
        m5_times = df_m5['time'].values
        m5_highs = df_m5['High'].values
        m5_lows = df_m5['Low'].values
        m5_opens = df_m5['Open'].values
        m5_closes = df_m5['Close'].values
    else:
        h1_highs = w_h1['High'].values
        h1_lows = w_h1['Low'].values
        h1_opens = w_h1['Open'].values

    # Grid Search Space
    entry_hours = list(range(8, 17))
    entry_mults = [0.5, 0.75, 1.0, 1.25, 1.5]
    sl_mults = [0.5, 1.0, 1.5]
    rr_mults = [1, 2, 3, 4]
    results = []
    
    for hr in entry_hours:
        entry_indices = np.where((h1_hours == hr) & (~np.isnan(avg_wicks)))[0]
        if len(entry_indices) == 0: continue
            
        for em in entry_mults:
            for slm in sl_mults:
                for rrm in rr_mults:
                    wins, losses, ambig = 0, 0, 0
                    pnl, peak, max_dd = 0.0, 0.0, 0.0
                    
                    for i in entry_indices:
                        entry_dist = avg_wicks[i] * em
                        if entry_dist <= 0: continue
                            
                        buy_limit = h1_closes[i] - entry_dist
                        sell_limit = h1_closes[i] + entry_dist
                        sl_dist = entry_dist * slm
                        tp_dist = sl_dist * rrm
                        
                        active_type, entry_price = 0, 0.0
                        
                        # ---------------------------------------------------------
                        # MODE 1: จำลองด้วยความแม่นยำสูง (M5 Data)
                        # ---------------------------------------------------------
                        if use_m5:
                            target_start_time = h1_times[i] + np.timedelta64(1, 'h')
                            start_idx = np.searchsorted(m5_times, target_start_time)
                            
                            # 1. Order Management (มองไปข้างหน้า 6 ชั่วโมง = 72 แท่ง M5)
                            act_idx = 0
                            for j in range(start_idx, min(len(m5_times), start_idx + 72)):
                                hit_b = m5_lows[j] <= buy_limit
                                hit_s = m5_highs[j] >= sell_limit
                                
                                if hit_b and hit_s:
                                    if m5_closes[j] > m5_opens[j]: active_type, entry_price, act_idx = 1, buy_limit, j
                                    else: active_type, entry_price, act_idx = -1, sell_limit, j
                                    break
                                elif hit_b: active_type, entry_price, act_idx = 1, buy_limit, j; break
                                elif hit_s: active_type, entry_price, act_idx = -1, sell_limit, j; break
                                    
                            # 2. Risk Management (มองไปข้างหน้า 72 ชั่วโมง = 864 แท่ง M5)
                            if active_type != 0:
                                resolved = False
                                for k in range(act_idx, min(len(m5_times), act_idx + 864)):
                                    hit_sl = (m5_lows[k] <= (entry_price - sl_dist)) if active_type == 1 else (m5_highs[k] >= (entry_price + sl_dist))
                                    hit_tp = (m5_highs[k] >= (entry_price + tp_dist)) if active_type == 1 else (m5_lows[k] <= (entry_price - tp_dist))
                                    
                                    if k == act_idx:
                                        if hit_sl: losses += 1; pnl -= 1.0; resolved = True; break
                                        elif hit_tp: ambig += 1; pnl += 0.25; resolved = True; break
                                    else:
                                        if hit_tp and hit_sl: ambig += 1; pnl += 0.25; resolved = True; break
                                        elif hit_tp: wins += 1; pnl += rrm; resolved = True; break
                                        elif hit_sl: losses += 1; pnl -= 1.0; resolved = True; break
                                        
                                if resolved:
                                    if pnl > peak: peak = pnl
                                    dd = peak - pnl
                                    if dd > max_dd: max_dd = dd

                        # ---------------------------------------------------------
                        # MODE 2: จำลองด้วย H1 ปกติ (กรณีไม่มี M5)
                        # ---------------------------------------------------------
                        else:
                            if i + 6 >= len(h1_closes): continue
                            act_idx = 0
                            
                            for j in range(1, 7):
                                hit_b = h1_lows[i+j] <= buy_limit
                                hit_s = h1_highs[i+j] >= sell_limit
                                if hit_b and hit_s:
                                    if h1_closes[i+j] > h1_opens[i+j]: active_type, entry_price, act_idx = 1, buy_limit, i+j
                                    else: active_type, entry_price, act_idx = -1, sell_limit, i+j
                                    break
                                elif hit_b: active_type, entry_price, act_idx = 1, buy_limit, i+j; break
                                elif hit_s: active_type, entry_price, act_idx = -1, sell_limit, i+j; break
                                    
                            if active_type != 0:
                                resolved = False
                                for k in range(act_idx, min(len(h1_closes), act_idx + 72)):
                                    hit_sl = (h1_lows[k] <= (entry_price - sl_dist)) if active_type == 1 else (h1_highs[k] >= (entry_price + sl_dist))
                                    hit_tp = (h1_highs[k] >= (entry_price + tp_dist)) if active_type == 1 else (h1_lows[k] <= (entry_price - tp_dist))
                                    
                                    if k == act_idx:
                                        if hit_sl: losses += 1; pnl -= 1.0; resolved = True; break
                                        elif hit_tp: ambig += 1; pnl += 0.25; resolved = True; break
                                    else:
                                        if hit_tp and hit_sl: ambig += 1; pnl += 0.25; resolved = True; break
                                        elif hit_tp: wins += 1; pnl += rrm; resolved = True; break
                                        elif hit_sl: losses += 1; pnl -= 1.0; resolved = True; break
                                        
                                if resolved:
                                    if pnl > peak: peak = pnl
                                    dd = peak - pnl
                                    if dd > max_dd: max_dd = dd
                                    
                    # สรุปผล
                    total_signals = wins + losses + ambig
                    if total_signals == 0: continue
                        
                    trades_per_week = total_signals / weeks_in_period
                    win_rate = (wins + (0.25 * ambig)) / total_signals
                    ev = pnl / total_signals
                    
                    results.append({
                        'Entry_Time': f"{hr:02d}:00",
                        'Entry_Mult': em,
                        'SL_Mult': slm,
                        'RR_Mult': rrm,
                        'Total_Trades': total_signals,
                        'Clean_Wins': wins,
                        'Clean_Losses': losses,
                        'Ambiguous': ambig,
                        'Win_Rate': win_rate,
                        'Trades/Week': trades_per_week,
                        'Max_DD': max_dd,
                        'EV': ev,
                        'Net_Profit_R': pnl
                    })
    return results

def process_uploaded_files(files):
    dfs = [pd.read_csv(file) for file in files]
    df = pd.concat(dfs, ignore_index=True)
    req_cols = ['time', 'Open', 'High', 'Low', 'Close']
    if not all(col in df.columns for col in req_cols): return None
    df['time'] = pd.to_datetime(df['time'])
    df.sort_values('time', inplace=True)
    df.drop_duplicates(subset=['time'], keep='first', inplace=True)
    return df

# ==========================================
# 3. ระบบหลังบ้าน (Execution & Display)
# ==========================================
st.title("🧪 Wick-Sweep Trap (Dual-Timeframe Optimizer)")
st.markdown("ระบบวิเคราะห์กลยุทธ์เชิงกลไก รองรับการจำลองความแม่นยำสูงด้วยข้อมูลระดับ 5 นาที (M5)")

if uploaded_h1 and run_button:
    with st.spinner("⏳ กำลังสกัด Feature และรัน Grid Search กว่า 540 รูปแบบ..."):
        
        # จัดการข้อมูล H1
        df_h1 = process_uploaded_files(uploaded_h1)
        if df_h1 is None:
            st.error("❌ ไฟล์ H1 ขาดคอลัมน์สำคัญ (time, Open, High, Low, Close)")
            st.stop()
            
        # สร้างฟีเจอร์ Wick-Sweep ให้ H1
        df_h1['Max_OC'] = df_h1[['Open', 'Close']].max(axis=1)
        df_h1['Min_OC'] = df_h1[['Open', 'Close']].min(axis=1)
        df_h1['Upper_Wick'] = df_h1['High'] - df_h1['Max_OC']
        df_h1['Lower_Wick'] = df_h1['Min_OC'] - df_h1['Low']
        df_h1['Wick_Sum'] = df_h1['Upper_Wick'] + df_h1['Lower_Wick']
        df_h1['X'] = df_h1['Wick_Sum'].rolling(3).sum()
        df_h1['Avg_Wick'] = df_h1['X'] / 6.0
        df_h1.dropna(inplace=True)

        # จัดการข้อมูล M5 (ถ้ามี)
        df_m5 = process_uploaded_files(uploaded_m5) if uploaded_m5 else None
        if df_m5 is not None:
            st.toast("✅ ตรวจพบข้อมูล M5: สลับเข้าสู่โหมด High-Precision Simulation")
        else:
            st.toast("⚠️ ไม่พบข้อมูล M5: รันด้วยโหมด H1 ปกติ")

        # กำหนดกำแพงเวลา
        sim_start = pd.to_datetime(sim_start_date)
        train_start = sim_start - timedelta(days=lookback_days)
        test_end = sim_start + timedelta(days=forward_days)
        train_weeks = lookback_days / 7.0
        test_weeks = forward_days / 7.0
        
        # รัน Optimizer
        train_results = optimize_wick_sweep(df_h1, df_m5, train_start, sim_start, train_weeks)
        
        if not train_results:
            st.warning("⚠️ ข้อมูลอดีตไม่เพียงพอ หรือไม่มีสัญญาณเกิดขี้น")
            st.stop()
            
        df_res = pd.DataFrame(train_results)
        df_filtered = df_res[(df_res['Max_DD'] <= target_max_dd) & (df_res['Trades/Week'] >= min_trades_per_week)].copy()
        df_filtered.sort_values(by='Net_Profit_R', ascending=False, inplace=True)
        df_filtered.reset_index(drop=True, inplace=True)

        # ==========================================
        # 4. Dashboard Output
        # ==========================================
        if df_filtered.empty:
            st.error("❌ **ไม่มีกลยุทธ์ใดรอดจากด่านอรหันต์ได้!** ลองปรับค่าลิมิตด้านซ้ายมือ")
        else:
            best = df_filtered.iloc[0]
            
            st.markdown("### 🏆 The Golden Setup (กลยุทธ์อันดับ 1 ที่พบในอดีต)")
            st.success(f"**วิธีการตั้ง Pending Order (นำไปตั้งค่าใน EA หรือเทรดมือ):**\n\n"
                       f"1. **รอให้ถึงเวลา:** `{best['Entry_Time']}` (เวลาตาม H1)\n"
                       f"2. **คำนวณระยะกางตาข่าย:** นำค่า Avg_Wick ล่าสุด มาคูณด้วย `{best['Entry_Mult']} เท่า`\n"
                       f"3. **วางออเดอร์:** ตั้ง Buy Limit และ Sell Limit ห่างจากราคา Close ตามระยะที่ได้\n"
                       f"4. **การบริหารความเสี่ยง:** ตั้ง SL ห่างจากจุดเข้า `{best['SL_Mult']} เท่า` ของระยะตาข่าย และตั้ง TP เอาไว้ `{best['RR_Mult']} เท่า` ของ SL\n"
                       f"5. **กฎ 6 ชั่วโมง:** หากออเดอร์ไม่ถูกเกี่ยวภายใน 6 ชั่วโมง ให้กดยกเลิกทิ้งทั้งหมด")
            
            st.markdown("---")
            st.markdown(f"### 🔮 พิสูจน์ด้วยข้อมูลอนาคต (Out-of-Sample: {sim_start.date()} ถึง {test_end.date()})")
            
            test_results = optimize_wick_sweep(df_h1, df_m5, sim_start, test_end, test_weeks)
            df_test = pd.DataFrame(test_results)
            
            if not df_test.empty:
                oos = df_test[(df_test['Entry_Time'] == best['Entry_Time']) & 
                              (df_test['Entry_Mult'] == best['Entry_Mult']) & 
                              (df_test['SL_Mult'] == best['SL_Mult']) & 
                              (df_test['RR_Mult'] == best['RR_Mult'])]
                
                if not oos.empty:
                    oos_best = oos.iloc[0]
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Win Rate", f"{oos_best['Win_Rate']:.1%}")
                    
                    ev_color = "normal" if oos_best['EV'] > 0 else "inverse"
                    c2.metric("Expected Value (EV)", f"{oos_best['EV']:.2f}R", delta="ผลตอบแทนต่อไม้", delta_color=ev_color)
                    
                    dd_color = "normal" if oos_best['Max_DD'] <= target_max_dd else "inverse"
                    c3.metric("Max Drawdown", f"-{oos_best['Max_DD']:.2f}R", delta="เช็คพอร์ตยุบ", delta_color=dd_color)
                    
                    pnl_color = "normal" if oos_best['Net_Profit_R'] > 0 else "inverse"
                    c4.metric("Net Profit (กำไรสุทธิ)", f"{oos_best['Net_Profit_R']:.2f}R", delta="อนาคตทำกำไรได้จริง", delta_color=pnl_color)
                else:
                    st.info("ตลาดอนาคตไม่มีสัญญาณเข้าเทรดด้วยเงื่อนไขนี้เลย (No Signals)")
            else:
                st.info("ตลาดอนาคตไม่มีสัญญาณเข้าเทรด (No Data/Signals)")

            st.markdown("---")
            st.markdown("### 📊 ตารางจัดอันดับผู้รอดชีวิตจาก Training Data (Sorted by Net Profit)")
            
            df_display = df_filtered.copy()
            df_display['Win_Rate'] = (df_display['Win_Rate'] * 100).map("{:.1f}%".format)
            df_display['EV'] = df_display['EV'].map("{:.2f}R".format)
            df_display['Max_DD'] = df_display['Max_DD'].map("-{:.2f}R".format)
            df_display['Net_Profit_R'] = df_display['Net_Profit_R'].map("{:.2f}R".format)
            df_display['Trades/Week'] = df_display['Trades/Week'].map("{:.1f}".format)
            
            st.dataframe(df_display, use_container_width=True)

elif not uploaded_h1:
    st.info("👆 **ระบบพร้อมใช้งาน:** อัปโหลดไฟล์ H1 (และ M5 ถ้ามี) กำหนดวันที่ แล้วรัน Optimizer ได้เลยครับ")
