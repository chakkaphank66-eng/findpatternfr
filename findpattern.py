import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta

# ==========================================
# 1. การตั้งค่าหน้าตา UI (Optimizer Mode)
# ==========================================
st.set_page_config(page_title="Gold H1 - Wick-Sweep Trap Optimizer", layout="wide")

st.sidebar.header("🕰️ Time Travel & Lab Optimizer")
st.sidebar.markdown("**Grid Search หาจุดเข้าเทรด Wick-Sweep Trap ที่ดีที่สุด**")

uploaded_files = st.sidebar.file_uploader("📂 อัปโหลดไฟล์ CSV หลายไฟล์", type=["csv"], accept_multiple_files=True)

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
# 2. ฟังก์ชันหลัก (Grid Search & Backtest Engine)
# ==========================================
def optimize_wick_sweep(df, start_date, end_date, weeks_in_period):
    """
    ฟังก์ชันแกนกลางสำหรับจำลอง Wick-Sweep Trap
    ประมวลผลด้วย Numpy Arrays เพื่อความเร็วระดับ Millisecond
    """
    # ดึงข้อมูลเฉพาะช่วงที่กำหนด
    mask = (df['time'] >= start_date) & (df['time'] < end_date)
    w_df = df[mask].reset_index(drop=True)
    
    if len(w_df) < 50:
        return []

    times = w_df['time'].dt.hour.values
    highs = w_df['High'].values
    lows = w_df['Low'].values
    closes = w_df['Close'].values
    opens = w_df['Open'].values
    avg_wicks = w_df['Avg_Wick'].values
    
    # กำหนดขอบเขตการค้นหา (Grid Search Space)
    entry_hours = list(range(8, 17)) # 08:00 - 16:00
    entry_mults = [0.5, 0.75, 1.0, 1.25, 1.5]
    sl_mults = [0.5, 1.0, 1.5]
    rr_mults = [1, 2, 3, 4]
    
    results = []
    
    for hr in entry_hours:
        # หา Index ของแท่งเทียนที่ตรงกับชั่วโมงเข้าเทรด
        entry_indices = np.where((times == hr) & (~np.isnan(avg_wicks)))[0]
        if len(entry_indices) == 0: continue
            
        for em in entry_mults:
            for slm in sl_mults:
                for rrm in rr_mults:
                    
                    wins, losses, ambig = 0, 0, 0
                    pnl, peak, max_dd = 0.0, 0.0, 0.0
                    
                    for i in entry_indices:
                        if i + 6 >= len(closes): continue # เผื่อระยะ 6 ชั่วโมงไม่พอ
                        
                        entry_dist = avg_wicks[i] * em
                        if entry_dist <= 0: continue
                            
                        buy_limit = closes[i] - entry_dist
                        sell_limit = closes[i] + entry_dist
                        sl_dist = entry_dist * slm
                        tp_dist = sl_dist * rrm
                        
                        active_type = 0 # 1=Buy, -1=Sell
                        entry_price = 0.0
                        act_idx = 0
                        
                        # 1. Order Management: สแกนไปข้างหน้า 6 ชั่วโมง หาจุดที่ Pending Order ถูกเกี่ยว
                        for j in range(1, 7):
                            hit_b = lows[i+j] <= buy_limit
                            hit_s = highs[i+j] >= sell_limit
                            
                            if hit_b and hit_s: break # ข้าม (แท่งสวิงแรงเกี่ยว 2 ฝั่งพร้อมกัน)
                            if hit_b:
                                active_type, entry_price, act_idx = 1, buy_limit, i+j
                                break
                            if hit_s:
                                active_type, entry_price, act_idx = -1, sell_limit, i+j
                                break
                                
                        # 2. Risk Management: เมื่อออเดอร์ Active ถือจนกว่าจะชน TP หรือ SL (มองไปข้างหน้าสูงสุด 72 ชม.)
                        if active_type != 0:
                            resolved = False
                            for k in range(act_idx, min(len(closes), act_idx + 72)):
                                if active_type == 1:
                                    hit_sl = lows[k] <= (entry_price - sl_dist)
                                    hit_tp = highs[k] >= (entry_price + tp_dist)
                                else:
                                    hit_sl = highs[k] >= (entry_price + sl_dist)
                                    hit_tp = lows[k] <= (entry_price - tp_dist)
                                    
                                # CRITICAL: Ambiguous Rule (ชนทั้งคู่ในแท่งเดียว)
                                if hit_tp and hit_sl:
                                    ambig += 1
                                    pnl += 0.25
                                    resolved = True
                                    break
                                elif hit_tp:
                                    wins += 1
                                    pnl += rrm
                                    resolved = True
                                    break
                                elif hit_sl:
                                    losses += 1
                                    pnl -= 1.0
                                    resolved = True
                                    break
                                    
                            if resolved:
                                if pnl > peak: peak = pnl
                                dd = peak - pnl
                                if dd > max_dd: max_dd = dd
                                
                    # 3. Evaluation Metrics สำหรับแต่ละ Pattern
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

# ==========================================
# 3. ระบบหลังบ้าน (Data Processing & Display)
# ==========================================
st.title("🧪 Wick-Sweep Trap (Hyper-Parameter Optimizer)")
st.markdown("ห้องแล็บส่วนตัวสำหรับการสแกนหากลยุทธ์เชิงกลไกที่สร้างกระแสเงินสดได้อย่างเสถียรที่สุด โดยปราศจาก Bias จาก AI")

if uploaded_files and run_button:
    with st.spinner("⏳ กำลังสกัด Feature และรัน Grid Search กว่า 540 รูปแบบ..."):
        
        # --- 3.1 อ่านและเตรียมข้อมูล ---
        dfs = [pd.read_csv(file) for file in uploaded_files]
        df = pd.concat(dfs, ignore_index=True)
        
        req_cols = ['time', 'Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in req_cols):
            st.error(f"❌ ขาดคอลัมน์สำคัญ: {', '.join(req_cols)}")
            st.stop()
            
        df['time'] = pd.to_datetime(df['time'])
        df.sort_values('time', inplace=True)
        df.drop_duplicates(subset=['time'], keep='first', inplace=True)
        
        # --- 3.2 Strategy Core Logic (Wick-Sweep Trap) ---
        df['Max_OC'] = df[['Open', 'Close']].max(axis=1)
        df['Min_OC'] = df[['Open', 'Close']].min(axis=1)
        df['Upper_Wick'] = df['High'] - df['Max_OC']
        df['Lower_Wick'] = df['Min_OC'] - df['Low']
        df['Wick_Sum'] = df['Upper_Wick'] + df['Lower_Wick']
        # X = Rolling sum ของ 3 แท่ง H1
        df['X'] = df['Wick_Sum'].rolling(3).sum()
        df['Avg_Wick'] = df['X'] / 6.0
        
        df.dropna(inplace=True)
        
        # --- 3.3 Data Slicing (กำแพงเวลา) ---
        sim_start = pd.to_datetime(sim_start_date)
        train_start = sim_start - timedelta(days=lookback_days)
        test_end = sim_start + timedelta(days=forward_days)
        
        train_weeks = lookback_days / 7.0
        test_weeks = forward_days / 7.0
        
        # รัน Optimizer บน Train Data (ข้อมูลอดีต)
        train_results = optimize_wick_sweep(df, train_start, sim_start, train_weeks)
        
        if not train_results:
            st.warning("⚠️ ข้อมูลไม่เพียงพอในช่วง Lookback หรือไม่มีสัญญาณเกิดขี้น")
            st.stop()
            
        # กรองข้อมูลด้วยด่านอรหันต์ (Strict Constraints)
        df_res = pd.DataFrame(train_results)
        df_filtered = df_res[(df_res['Max_DD'] <= target_max_dd) & (df_res['Trades/Week'] >= min_trades_per_week)].copy()
        df_filtered.sort_values(by='Net_Profit_R', ascending=False, inplace=True)
        df_filtered.reset_index(drop=True, inplace=True)

        # ==========================================
        # 4. Dashboard Output (รายงานผลการสแกน)
        # ==========================================
        if df_filtered.empty:
            st.error("❌ **ไม่มีกลยุทธ์ใดรอดจากด่านอรหันต์ได้!**\n\n*คำแนะนำ: ลองเพิ่ม Max Drawdown หรือลดจำนวนขั้นต่ำ Trades/Week เพื่อคลายความตึงเครียดของระบบ*")
        else:
            best = df_filtered.iloc[0]
            
            st.markdown("### 🏆 The Golden Setup (กลยุทธ์อันดับ 1 ที่พบในอดีต)")
            st.success(f"**วิธีการตั้ง Pending Order (นำไปตั้งค่าใน EA หรือเทรดมือ):**\n\n"
                       f"1. **รอให้ถึงเวลา:** `{best['Entry_Time']}` (เวลาตาม CSV)\n"
                       f"2. **คำนวณระยะกางตาข่าย:** นำค่า Avg_Wick ล่าสุด มาคูณด้วย `{best['Entry_Mult']} เท่า`\n"
                       f"3. **วางออเดอร์:** ตั้ง Buy Limit และ Sell Limit ห่างจากราคา Close ตามระยะที่ได้\n"
                       f"4. **การบริหารความเสี่ยง:** ตั้ง SL ห่างจากจุดเข้า `{best['SL_Mult']} เท่า` ของระยะตาข่าย และตั้ง TP เอาไว้ `{best['RR_Mult']} เท่า` ของ SL\n"
                       f"5. **กฎ 6 ชั่วโมง:** หากออเดอร์ไม่ถูกเกี่ยวภายใน 6 ชั่วโมง ให้กดยกเลิกทิ้งทั้งหมด")
            
            # รันแบบ Out-of-Sample (พิสูจน์ในอนาคต) ด้วย Golden Setup
            st.markdown("---")
            st.markdown(f"### 🔮 พิสูจน์ด้วยข้อมูลอนาคต (Out-of-Sample: {sim_start.date()} ถึง {test_end.date()})")
            st.markdown("*เราล็อกพารามิเตอร์อันดับ 1 มาเทรดในอนาคตที่ระบบไม่เคยเห็น เพื่อดูว่า 'ของจริง' รอดหรือไม่*")
            
            test_results = optimize_wick_sweep(df, sim_start, test_end, test_weeks)
            df_test = pd.DataFrame(test_results)
            
            if not df_test.empty:
                # ดึงเฉพาะผลลัพธ์ของ Golden Setup ในข้อมูล Test
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
                    c3.metric("Max Drawdown", f"-{oos_best['Max_DD']:.2f}R", delta="เช็คว่าพอร์ตยุบเกินลิมิตไหม", delta_color=dd_color)
                    
                    pnl_color = "normal" if oos_best['Net_Profit_R'] > 0 else "inverse"
                    c4.metric("Net Profit (กำไรสุทธิ)", f"{oos_best['Net_Profit_R']:.2f}R", delta="ทำกำไรในอนาคตได้จริง", delta_color=pnl_color)
                else:
                    st.info("ตลาดอนาคตไม่มีสัญญาณเข้าเทรดด้วยเงื่อนไขนี้เลย (No Signals)")
            else:
                st.info("ตลาดอนาคตไม่มีสัญญาณเข้าเทรด (No Data/Signals)")

            # ตารางสถิติผู้รอดชีวิตจากช่วง Train Data
            st.markdown("---")
            st.markdown("### 📊 ตารางจัดอันดับกลยุทธ์ที่รอดชีวิตจาก Training Data (Sorted by Net Profit)")
            st.markdown("*เฉพาะรูปแบบที่จำนวนเทรด > ลิมิต และ Max DD < ลิมิต*")
            
            # ฟอร์แมตตัวเลขให้อ่านง่าย
            df_display = df_filtered.copy()
            df_display['Win_Rate'] = (df_display['Win_Rate'] * 100).map("{:.1f}%".format)
            df_display['EV'] = df_display['EV'].map("{:.2f}R".format)
            df_display['Max_DD'] = df_display['Max_DD'].map("-{:.2f}R".format)
            df_display['Net_Profit_R'] = df_display['Net_Profit_R'].map("{:.2f}R".format)
            df_display['Trades/Week'] = df_display['Trades/Week'].map("{:.1f}".format)
            
            st.dataframe(df_display, use_container_width=True)

elif not uploaded_files:
    st.info("👆 **ระบบพร้อมใช้งาน:** ลากไฟล์ CSV มาวาง กรอกวันที่ และกด Run Full Optimizer ได้เลยครับ")
