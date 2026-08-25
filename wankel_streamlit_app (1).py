import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

# Set page configuration
st.set_page_config(
    page_title="Wankel Engine 0D Simulator",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🌀 Wankel Rotary Engine 0D Thermodynamic Simulator")
st.markdown("""
A modern, interactive graphical interface for Wankel engine design and cycle analysis.
Grounded on **Kenichi Yamamoto's** geometric relations and **Timothy John Norman's** MIT MSc Thesis (1983).
""")

class WankelSimulator0D:
    def __init__(self, R=0.105, e=0.015, b=0.080, r_c=9.6, N=3000.0, 
                 p_in=1.013, T_in=310.0, p_ex=1.05, T_wall=420.0,
                 LHV=44e6, AFR=14.7, A_leak=1.5e-6, V_pocket_cc=0.0):
        self.R = R
        self.e = e
        self.b = b
        self.r_c = r_c
        self.N = N
        self.omega = N * (2 * np.pi / 60.0) # shaft angular velocity (rad/s)
        
        self.p_in = p_in * 1e5   # Pa
        self.T_in = T_in         # K
        self.p_ex = p_ex * 1e5   # Pa
        self.T_wall = T_wall     # K
        self.LHV = LHV
        self.AFR = AFR
        self.A_leak = A_leak
        self.V_pocket = V_pocket_cc * 1e-6 # m^3
        
        # Gas properties (approximate for air-fuel mixture)
        self.R_gas = 287.05      # J/kg.K
        self.gamma = 1.35        # average ratio of specific heats
        self.Cv = self.R_gas / (self.gamma - 1)
        self.Cp = self.gamma * self.Cv
        
        # Derived geometry
        self.V_d = 3.0 * np.sqrt(3) * self.e * self.R * self.b
        self.V_c = self.V_d / (self.r_c - 1.0)
        self.V_max = self.V_c + self.V_d
        
    def get_volume(self, theta_deg):
        theta_rad = np.radians(theta_deg)
        V = self.V_c + (self.V_d / 2.0) * (1.0 - np.cos(2.0 * theta_rad / 3.0)) + self.V_pocket
        dV_dtheta_rad = (self.V_d / 3.0) * np.sin(2.0 * theta_rad / 3.0)
        dV_dtheta = dV_dtheta_rad * (np.pi / 180.0) # per degree
        return V, dV_dtheta

    def get_surface_area(self, V):
        A_sides = 2.0 * (V / self.b)
        A_rotor = self.b * (np.sqrt(3) * self.R)
        A_housing = self.b * (np.sqrt(3) * self.R * (1.0 + 1.5 * (self.e / self.R)))
        return A_sides + A_rotor + A_housing

    def woschni_norman_ht(self, p, T, V):
        B = self.R
        p_bar = p / 1e5
        V_m = np.pi * self.N * self.R / 90.0
        w = 1.0 * V_m
        h_c = 3.26 * (B ** -0.2) * (p_bar ** 0.8) * (T ** -0.53) * (w ** 0.8)
        return h_c

    def wiebe_combustion(self, theta, theta_start=540.0, theta_dur=50.0, m=2.0, a=5.0):
        if theta < theta_start:
            return 0.0, 0.0
        elif theta > (theta_start + theta_dur):
            return 1.0, 0.0
        else:
            y = (theta - theta_start) / theta_dur
            x_b = 1.0 - np.exp(-a * (y ** (m + 1)))
            dx_b_dtheta = (a * (m + 1) / theta_dur) * (y ** m) * np.exp(-a * (y ** (m + 1)))
            return x_b, dx_b_dtheta

    def simulate(self, d_theta=0.5):
        thetas = np.arange(0.0, 1080.0 + d_theta, d_theta)
        
        pressures = np.zeros_like(thetas)
        temperatures = np.zeros_like(thetas)
        volumes = np.zeros_like(thetas)
        masses = np.zeros_like(thetas)
        heat_release_rate = np.zeros_like(thetas)
        heat_loss_rate = np.zeros_like(thetas)
        cum_heat_release = np.zeros_like(thetas)
        cum_heat_loss = np.zeros_like(thetas)
        work = 0.0
        
        V0, _ = self.get_volume(0.0)
        pressures[0] = self.p_in
        temperatures[0] = self.T_in
        volumes[0] = V0
        masses[0] = (self.p_in * V0) / (self.R_gas * self.T_in)
        
        m_air_trapped_est = (self.p_in * self.V_max) / (self.R_gas * self.T_in)
        m_fuel = m_air_trapped_est / self.AFR
        Q_total = m_fuel * self.LHV
        
        dt_dtheta = 1.0 / (6.0 * self.N)
        
        for i in range(len(thetas) - 1):
            theta = thetas[i]
            p = pressures[i]
            T = temperatures[i]
            m = masses[i]
            V, dV = self.get_volume(theta)
            volumes[i] = V
            
            V_next, dV_next = self.get_volume(theta + d_theta)
            
            inlet_flow = 0.0
            exhaust_flow = 0.0
            dQ_comb = 0.0
            
            if 0.0 <= theta < 270.0:
                p_target = self.p_in
                T_target = self.T_in
                m_next = (p_target * V_next) / (self.R_gas * T_target)
                inlet_flow = (m_next - m) / d_theta
            elif 810.0 <= theta <= 1080.0:
                p_target = self.p_ex
                m_next = (p_target * V_next) / (self.R_gas * T)
                exhaust_flow = (m_next - m) / d_theta
            else:
                # Closed Cycle
                if 540.0 <= theta < 810.0:
                    x_b, dx_b = self.wiebe_combustion(theta, theta_start=540.0, theta_dur=50.0)
                    dQ_comb = Q_total * dx_b
                    heat_release_rate[i] = dQ_comb / dt_dtheta
                
                h_c = self.woschni_norman_ht(p, T, V)
                A_wall = self.get_surface_area(V)
                dQ_ht_dt = h_c * A_wall * (T - self.T_wall)
                dQ_ht = dQ_ht_dt * dt_dtheta
                heat_loss_rate[i] = dQ_ht_dt
                
                if p > self.p_in:
                    pr = self.p_in / p
                    gamma_term = (2.0 / (self.gamma + 1.0)) ** ((self.gamma + 1.0) / (self.gamma - 1.0))
                    m_dot_leak = self.A_leak * p * np.sqrt((self.gamma / (self.R_gas * T)) * gamma_term)
                    dm_leak = m_dot_leak * dt_dtheta
                    m_next = m - dm_leak
                else:
                    m_next = m
                
                dV_step = V_next - V
                dT = (dQ_comb * d_theta - dQ_ht * d_theta - p * dV_step) / (m * self.Cv)
                T_next = T + dT
                p_next = (m_next * self.R_gas * T_next) / V_next
                
                work += p * dV_step
                
                pressures[i+1] = p_next
                temperatures[i+1] = T_next
                masses[i+1] = m_next
                cum_heat_release[i+1] = cum_heat_release[i] + dQ_comb * d_theta
                cum_heat_loss[i+1] = cum_heat_loss[i] + dQ_ht * d_theta
                continue
                
            pressures[i+1] = p_target
            temperatures[i+1] = T_target if 0.0 <= theta < 270.0 else T
            masses[i+1] = m + (inlet_flow + exhaust_flow) * d_theta
            cum_heat_release[i+1] = cum_heat_release[i]
            cum_heat_loss[i+1] = cum_heat_loss[i]
            
            dV_step = V_next - V
            work += pressures[i] * dV_step

        volumes[-1] = self.get_volume(thetas[-1])[0]
        
        W_ind = work
        P_ind = W_ind * (self.N / 60.0) / 1000.0
        eta_ind = (W_ind / Q_total) * 100.0 if Q_total > 0 else 0.0
        imep = (W_ind / self.V_d) / 1e5
        
        idx_270 = np.argmin(np.abs(thetas - 270.0))
        idx_540 = np.argmin(np.abs(thetas - 540.0))
        m_270 = masses[idx_270]
        m_540 = masses[idx_540]
        leakage_loss = (1.0 - m_540 / m_270) * 100.0 if m_270 > 0 else 0.0
        
        # Volumetric efficiency based on fresh trapped air
        m_fresh_trapped = m_540 - masses[0]
        ideal_air_swept = (self.p_in * self.V_d) / (self.R_gas * self.T_in)
        vol_eff = (m_fresh_trapped / ideal_air_swept) * 100.0 if ideal_air_swept > 0 else 0.0
        
        results = pd.DataFrame({
            'theta': thetas,
            'volume_cm3': volumes * 1e6,
            'pressure_bar': pressures / 1e5,
            'temperature_K': temperatures,
            'mass_g': masses * 1e3,
            'heat_release_kW': heat_release_rate / 1000.0,
            'heat_loss_kW': heat_loss_rate / 1000.0,
            'cum_heat_release_J': cum_heat_release,
            'cum_heat_loss_J': cum_heat_loss
        })
        
        summary = {
            'V_d_cm3': self.V_d * 1e6,
            'V_c_cm3': self.V_c * 1e6,
            'compression_ratio': self.r_c,
            'indicated_work_J': W_ind,
            'indicated_power_kW': P_ind,
            'indicated_efficiency_pct': eta_ind,
            'imep_bar': imep,
            'volumetric_efficiency_pct': vol_eff,
            'total_heat_released_J': Q_total,
            'total_heat_loss_J': cum_heat_loss[-1],
            'leakage_loss_pct': leakage_loss
        }
        
        return results, summary

# Sidebar - Geometry
st.sidebar.header("🔧 Engine Geometry")
R_mm = st.sidebar.slider("Rotor Generating Radius R (mm)", 50.0, 200.0, 105.0, step=1.0)
e_mm = st.sidebar.slider("Eccentricity e (mm)", 5.0, 30.0, 15.0, step=0.5)
b_mm = st.sidebar.slider("Rotor Width b (mm)", 20.0, 150.0, 80.0, step=1.0)
r_c = st.sidebar.slider("Compression Ratio r_c (-)", 5.0, 15.0, 9.6, step=0.1)
V_pocket_cc = st.sidebar.slider("Rotor Pocket Volume (cc)", 0.0, 100.0, 20.0, step=1.0)

# Sidebar - Operating Conditions
st.sidebar.header("⚙️ Operating Conditions")
N_rpm = st.sidebar.slider("Engine Speed N (RPM)", 1000, 8000, 3000, step=250)
T_wall_k = st.sidebar.slider("Wall Temperature T_wall (K)", 300, 600, 420, step=10)
A_leak_mm2 = st.sidebar.slider("Equivalent Leakage Area (mm²)", 0.0, 5.0, 1.5, step=0.1)

# Sidebar - Combustion & Fuel
st.sidebar.header("🔥 Combustion & Fuel")
AFR = st.sidebar.slider("Air-Fuel Ratio AFR (-)", 10.0, 20.0, 14.7, step=0.1)
ignition_deg = st.sidebar.slider("Ignition Spark Timing (°SA)", 500.0, 580.0, 540.0, step=1.0)
comb_dur_deg = st.sidebar.slider("Combustion Duration (°)", 20.0, 90.0, 50.0, step=5.0)

# Convert to SI units
R = R_mm * 1e-3
e = e_mm * 1e-3
b = b_mm * 1e-3
A_leak = A_leak_mm2 * 1e-6

# Instantiate and simulate
sim = WankelSimulator0D(
    R=R, e=e, b=b, r_c=r_c, N=N_rpm,
    T_wall=T_wall_k, A_leak=A_leak, AFR=AFR,
    V_pocket_cc=V_pocket_cc
)

results, summary = sim.simulate()

# Layout of the Main Page tabs
tab_dash, tab_data, tab_theory = st.tabs(["📊 Performance Dashboard", "📂 Raw Simulation Data", "📝 Theoretical Models"])

with tab_dash:
    # Key Performance Indicator (KPI) Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Indicated Power", f"{summary['indicated_power_kW']:.2f} kW", help="Total indicated power generated by 1 rotor (3 active chambers)")
    col2.metric("Thermal Efficiency", f"{summary['indicated_efficiency_pct']:.2f} %")
    col3.metric("IMEP", f"{summary['imep_bar']:.2f} bar", help="Indicated Mean Effective Pressure")
    col4.metric("Volumetric Efficiency", f"{summary['volumetric_efficiency_pct']:.2f} %")
    col5.metric("Blow-by Leakage Loss", f"{summary['leakage_loss_pct']:.2f} %")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        # P-V Diagram Plot
        st.subheader("Pressure-Volume (P-V) Diagram")
        log_scale = st.checkbox("Logarithmic Scale")
        
        fig_pv, ax_pv = plt.subplots(figsize=(6, 4))
        ax_pv.plot(results['volume_cm3'], results['pressure_bar'], 'r-', linewidth=2.5)
        ax_pv.set_xlabel("Chamber Volume ($cm^3$)")
        ax_pv.set_ylabel("Pressure (bar)")
        ax_pv.grid(True, linestyle="--", alpha=0.5)
        
        if log_scale:
            ax_pv.set_xscale('log')
            ax_pv.set_yscale('log')
            ax_pv.set_title("Log P - Log V Cycle Diagram", fontweight="bold")
        else:
            ax_pv.set_title("Thermodynamic P-V Cycle Diagram", fontweight="bold")
            
        st.pyplot(fig_pv)

    with col_right:
        # Energy and Heat Plot
        st.subheader("Energy Balance & Heat Release")
        fig_en, ax_en = plt.subplots(figsize=(6, 4))
        ax_en.plot(results['theta'], results['cum_heat_release_J'], 'm-', label='Heat Released', linewidth=2)
        ax_en.plot(results['theta'], results['cum_heat_loss_J'], 'k--', label='Heat Loss to Walls', linewidth=2)
        ax_en.set_xlabel("Crank Angle (deg)")
        ax_en.set_ylabel("Energy (J)")
        ax_en.set_title("Cumulative Heat Release & Wall Heat Loss", fontweight="bold")
        ax_en.legend()
        ax_en.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig_en)

    st.markdown("---")
    
    # 2x2 multiplots
    st.subheader("In-Cylinder Parameter Variations vs. Crank Angle")
    fig_multi, axs = plt.subplots(2, 2, figsize=(12, 8))
    
    # Pressure
    axs[0, 0].plot(results['theta'], results['pressure_bar'], 'b-', linewidth=2)
    axs[0, 0].set_title("In-Cylinder Pressure", fontweight="bold")
    axs[0, 0].set_xlabel("Crank Angle (deg)")
    axs[0, 0].set_ylabel("Pressure (bar)")
    axs[0, 0].grid(True, linestyle="--", alpha=0.5)
    
    # Temperature
    axs[0, 1].plot(results['theta'], results['temperature_K'], 'r-', linewidth=2)
    axs[0, 1].set_title("In-Cylinder Temperature", fontweight="bold")
    axs[0, 1].set_xlabel("Crank Angle (deg)")
    axs[0, 1].set_ylabel("Temperature (K)")
    axs[0, 1].grid(True, linestyle="--", alpha=0.5)
    
    # Volume
    axs[1, 0].plot(results['theta'], results['volume_cm3'], 'g-', linewidth=2)
    axs[1, 0].set_title("Chamber Volume", fontweight="bold")
    axs[1, 0].set_xlabel("Crank Angle (deg)")
    axs[1, 0].set_ylabel("Volume ($cm^3$)")
    axs[1, 0].grid(True, linestyle="--", alpha=0.5)
    
    # Mass
    axs[1, 1].plot(results['theta'], results['mass_g'], 'y-', linewidth=2)
    axs[1, 1].set_title("In-Cylinder Gas Mass", fontweight="bold")
    axs[1, 1].set_xlabel("Crank Angle (deg)")
    axs[1, 1].set_ylabel("Gas Mass (g)")
    axs[1, 1].grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    st.pyplot(fig_multi)

with tab_data:
    st.subheader("Simulation Results Data Table")
    st.markdown("Step resolution: 0.5 degrees. Total rows: 2161.")
    
    # Table display
    st.dataframe(results, height=400)
    
    # CSV download
    csv = results.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Raw Simulation Data (CSV)",
        data=csv,
        file_name="wankel_0d_results.csv",
        mime="text/csv"
    )

with tab_theory:
    st.subheader("Theoretical Equations & Modeling Details")
    st.markdown("""
    ### 1. Engine Geometry
    The volume of the combustion chamber $V(\\theta)$ as a function of output shaft angle $\theta$ is modeled using **Kenichi Yamamoto's** classic epitrochoidal equations:
    $$V(\\theta) = V_c + \\frac{V_d}{2}\\left(1 - \\cos\\left(\\frac{2}{3}\\theta\\right)\\right) + V_{pocket}$$
    Where:
    - $V_d$ is the swept volume of a single rotor face: $V_d = 3\\sqrt{3} e R b$
    - $V_c$ is the clearance volume (combustion pocket + clearance)
    - $e$ is eccentricity, $R$ is generating radius, and $b$ is housing width.

    ### 2. Convective Heat Transfer
    We use the modified **Woschni-Norman correlation** for Wankel engines. Timothy Norman's MIT thesis adjusted the characteristic gas velocity using the mean linear rotor tip speed $V_m$:
    $$V_m = \\frac{\\pi N R}{90}$$
    The heat transfer coefficient is calculated as:
    $$h_c = 3.26 B^{-0.2} P^{0.8} T^{-0.53} w^{0.8}$$
    Where $w$ is the characteristic gas velocity set equal to $V_m$.

    ### 3. Gas Leakage (Blow-by)
    Gas leakage across the apex seals is modeled as isentropic orifice flow of a compressible fluid:
    $$\\dot{m}_{leak} = A_{leak} P \\sqrt{\\frac{\\gamma}{R_{gas} T} \\left(\\frac{2}{\\gamma+1}\\right)^{\\frac{\\gamma+1}{\\gamma-1}}}$$
    This acts as a significant loss mechanism at lower engine speeds.
    """)
