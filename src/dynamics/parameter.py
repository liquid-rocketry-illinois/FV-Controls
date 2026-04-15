from sympy import *
import numpy as np
import pandas as pd
from typing import Callable

class Parameter():  

    def setRocketParams(self, I_0: float, I_f: float, I_3: float,
                        I_3_f: float,
                        x_CG_0: float, x_CG_f: float,
                        m_0: float, m_f: float, m_p: float,
                        d: float, L_ne: float,
                        t_launch_rail_clearance: float, t_motor_burnout: float, t_estimated_apogee: float,
                        C_d: float = None, Cnalpha_rocket: float = None,
                        CP_func: Callable[[Expr], Expr] = None):  # replaced by 2D CP_func_2d (AoA + Mach)
        """Set the rocket parameters.

        Args:
            I_0 (float): Initial moment of inertia in kg·m².
            I_f (float): Final moment of inertia in kg·m².
            I_3 (float): Initial moment of inertia about the z-axis in kg·m².
            I_3_f (float): Burnout moment of inertia about the z-axis in kg·m².
            x_CG_0 (float): Initial center of gravity location in meters.
            x_CG_f (float): Final center of gravity location in meters.
            m_0 (float): Initial mass in kg.
            m_f (float): Final mass in kg.
            m_p (float): Propellant mass in kg.
            d (float): Rocket diameter in meters.
            L_ne (float): Length from nose to engine exit in meters.
            Cnalpha_rocket (float): Rocket normal force coefficient derivative.
            t_motor_burnout (float): Time until motor burnout in seconds.
            t_estimated_apogee (float, optional): Estimated time until apogee in seconds.
            t_launch_rail_clearance (float): Time until launch rail clearance in seconds.
            CP_func (Callable[[Expr], Expr]): User defined function of center of pressure location as a function of angle of attack (deg).\
                Takes Expr 'AoA_deg' as parameter. Returns function fit equation of center of pressure location vs AoA (e.g. using Google Sheets).
        """
        self.I_0 = I_0
        self.I_f = I_f
        self.I_3 = I_3
        self.I_3_f = I_3_f
        self.x_CG_0 = x_CG_0
        self.x_CG_f = x_CG_f
        self.m_0 = m_0
        self.m_f = m_f
        self.m_p = m_p
        self.d = d
        self.L_ne = L_ne
        self.C_d = C_d
        self.Cnalpha_rocket = Cnalpha_rocket
        self.t_launch_rail_clearance = t_launch_rail_clearance
        self.t_motor_burnout = t_motor_burnout
        self.t_estimated_apogee = t_estimated_apogee
        self.CP_func = CP_func
    
    def setFinParams(self, N: int, Cr: float, Ct: float, s: float, delta: float, Cnalpha_fin: float = None):
        """Set the main fin parameters used in roll moment equations.

        Args:
            N (int): Number of fins.
            Cr (float): Fin root chord in meters.
            Ct (float): Fin tip chord in meters.
            s (float): Fin exposed semi-span (body surface to tip) in meters.
            delta (float): Fin cant angle in degrees.
            Cnalpha_fin (float): Per-fin normal force coefficient derivative (1/rad).
                If None, compute via compute_cnalpha_barrowman() after setMainFinGeometry().
        """
        self.N = N
        self.Cr = Cr
        self.Ct = Ct
        self.s = s
        self.Cnalpha_fin = Cnalpha_fin
        self.delta = delta

    # ------------------------------------------------------------------ #
    #  Barrowman geometry setters                                          #
    # ------------------------------------------------------------------ #

    def setNoseGeometry(self, L_nose: float, R_nose: float = None,
                        shape: str = 'conical'):
        """Set nose cone geometry for Barrowman CN_alpha computation.

        Args:
            L_nose (float): Nose cone length in meters (tip to base).
            R_nose (float): Nose cone base radius in meters.
                Defaults to d/2 (full-diameter nose) when None.
            shape (str): Nose cone shape. One of:
                'conical', 'ogive', 'von_karman', 'elliptical', 'parabolic'.
        """
        self.L_nose   = L_nose
        self.R_nose   = R_nose   # None → uses d/2 at compute time
        self.nose_shape = shape

    def setCanardGeometry(self, N: int, Cr: float, Ct: float, s: float,
                          x_LE: float, R_body: float,
                          x_sweep: float = 0.0):
        """Set canard fin-set geometry for Barrowman CN_alpha computation.

        The canard CN_alpha contributes to the total rocket CN_alpha.
        For roll-moment calculations the relevant fin set is the main fins
        (setMainFinGeometry / setFinParams), not the canards.

        Args:
            N (int): Number of canard fins.
            Cr (float): Root chord in meters.
            Ct (float): Tip chord in meters.
            s (float): Exposed semi-span from body surface to tip (m).
            x_LE (float): Axial position of canard root leading edge from nose tip (m).
            R_body (float): Body radius at canard root chord (m).
            x_sweep (float): Axial offset of tip leading edge behind root leading edge (m).
                Zero for unswept canards.
        """
        self.N_canards          = N
        self.Cr_canards         = Cr
        self.Ct_canards         = Ct
        self.s_canards          = s
        self.x_canard_LE        = x_LE
        self.R_body_at_canard   = R_body
        self.x_sweep_canards    = x_sweep

    def setMainFinGeometry(self, x_LE: float, R_body: float = None,
                           x_sweep: float = 0.0):
        """Set axial position and sweep of the main fin set for Barrowman computation.

        Must be called AFTER setFinParams (which provides N, Cr, Ct, s).

        Args:
            x_LE (float): Axial position of main fin root leading edge from nose tip (m).
            R_body (float): Body radius at fin root chord (m). Defaults to d/2 when None.
            x_sweep (float): Axial offset of fin tip leading edge behind fin root
                leading edge (m). Equals RocketPy sweep_length parameter.
        """
        self.x_fin_LE       = x_LE
        self.R_body_at_fin  = R_body
        self.x_sweep_fin    = x_sweep

    def setTailGeometry(self, top_radius: float, bottom_radius: float,
                        length: float, x_position: float):
        """Set boattail/shoulder geometry for Barrowman CN_alpha computation.

        Models a conical frustum (diameter transition). For a typical boattail
        (top_radius > bottom_radius) the contribution is slightly destabilizing
        (negative CN_alpha), which is the physically correct behaviour.

        Args:
            top_radius (float): Forward (larger) radius of the transition in meters.
            bottom_radius (float): Aft (smaller) radius of the transition in meters.
            length (float): Axial length of the transition in meters.
            x_position (float): Axial position of the forward face of the transition
                from the nose tip (m).
        """
        self.tail_type       = 'boattail'
        self.R_boattail_fore = top_radius
        self.R_boattail_aft  = bottom_radius
        self.L_boattail      = length
        self.x_boattail      = x_position

    # ------------------------------------------------------------------ #
    #  Barrowman static helpers                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _barrowman_nose(L_nose: float, R_nose: float, R_ref: float,
                        shape: str) -> tuple:
        """Barrowman CN_alpha and CP for a nose cone (slender-body theory).

        Derivation (Barrowman 1967):
            Normal force per unit length ∝ d(r²)/dx, where r(x) is the local radius.
            Integrating from tip (x=0) to base (x=L_nose) yields CN_alpha = 2*(A_N/A_ref).
            CP_x is the centroid of the d(r²)/dx distribution, which is shape-dependent.

        CP fractions (from nose tip, as fraction of L_nose):
            conical   : 2/3   (from ∫x·x dx / ∫x dx, r ∝ x)
            ogive     : 0.534 (Barrowman 1967, tangent ogive lookup)
            von_karman: 0.500 (LD-Haack series C=0, numerical approximation)
            elliptical: 1/3   (half-ellipse, pressure weighted toward tip)
            parabolic : 0.500 (parabolic profile)

        Args:
            L_nose: Nose length (m).
            R_nose: Nose base radius (m).
            R_ref: Reference body radius (m).
            shape: Nose shape string (see above).

        Returns:
            (CN_alpha, x_CP) — CN_alpha in 1/rad, x_CP in metres from nose tip.
        """
        CN_alpha = 2.0 * (R_nose / R_ref) ** 2

        _cp_frac = {
            'conical':    2.0 / 3.0,
            'ogive':      0.534,
            'von_karman': 0.500,
            'elliptical': 1.0 / 3.0,
            'parabolic':  0.500,
            'power':      0.500,
        }
        frac = _cp_frac.get(shape.lower().replace(' ', '_').replace('-', '_'), 2.0 / 3.0)
        x_CP = frac * L_nose

        return CN_alpha, x_CP

    @staticmethod
    def _barrowman_finset(N: int, Cr: float, Ct: float, s: float,
                          R_body: float, R_ref: float,
                          x_LE: float, x_sweep: float = 0.0) -> tuple:
        """Barrowman CN_alpha and CP for a trapezoidal fin set.

        Implements the Barrowman (1967) finite-span fin equations:

        Formulae:
            d_ref = 2 · R_ref                    (reference diameter)
            K_fb  = 1 + R_body / (R_body + s)   (body-fin interference factor)

            Midchord line length (geometrically exact):
                The midchord line connects the midpoint of the root chord to
                the midpoint of the tip chord.
                  Root midchord axial offset from root LE: Cr/2
                  Tip  midchord axial offset from root LE: x_sweep + Ct/2
                  Axial span of midchord line: x_sweep + Ct/2 - Cr/2
                                             = x_sweep - (Cr - Ct)/2

                l_m = sqrt(s² + (x_sweep - (Cr - Ct)/2)²)

            Barrowman normal-force coefficient derivative (normalised to
            A_ref = π R_ref², the body cross-sectional area):

                CN_alpha_per_fin = K_fb · 4·(s / d_ref)²
                                   ─────────────────────────────────────────
                                   1 + sqrt(1 + (2 l_m / (Cr + Ct))²)

                CN_alpha_total   = N · CN_alpha_per_fin

            Centre of pressure (axial, from nose tip):
                x_CP = x_LE + x_sweep/3 + Cr·(Cr + 2·Ct) / (3·(Cr + Ct))

        Note on normalisation:
            The formula 4·(s/d_ref)² = s²/R_ref² is the Barrowman convention.
            Combined with A_ref = π·R_ref² in the force equation the physical
            normal force is N · K_fb · π · s² · q / denominator.
            This matches RocketPy's CN_alpha output for fin sets.

        Args:
            N: Number of fins.
            Cr: Root chord (m).
            Ct: Tip chord (m).
            s: Exposed semi-span from body surface to fin tip (m).
            R_body: Body radius at fin root (m).
            R_ref: Reference body radius (m).
            x_LE: Axial position of fin root leading edge from nose tip (m).
            x_sweep: Axial offset of fin tip leading edge behind fin root
                leading edge (m). Equals RocketPy sweep_length. Zero for
                unswept fins.

        Returns:
            (CN_alpha_total, CN_alpha_per_fin, x_CP)
            CN_alphas in 1/rad (normalised to A_ref = π R_ref²),
            x_CP in metres from nose tip.
        """
        K_fb  = 1.0 + R_body / (R_body + s)
        d_ref = 2.0 * R_ref

        # Midchord line: axial span = x_sweep − (Cr−Ct)/2
        # (x_sweep is tip-LE to root-LE offset; taper shifts tip midchord forward)
        midchord_axial = x_sweep - (Cr - Ct) / 2.0
        l_m = np.sqrt(s ** 2 + midchord_axial ** 2)

        CN_alpha_per_fin = K_fb * (4.0 * (s / d_ref) ** 2) / (
            1.0 + np.sqrt(1.0 + (2.0 * l_m / (Cr + Ct)) ** 2)
        )
        CN_alpha_total = N * CN_alpha_per_fin

        # Axial CP from root LE (Barrowman chord-weighted centroid + 1/3 sweep)
        x_cp_from_LE = x_sweep / 3.0 + Cr * (Cr + 2.0 * Ct) / (3.0 * (Cr + Ct))
        x_CP = x_LE + x_cp_from_LE

        return CN_alpha_total, CN_alpha_per_fin, x_CP

    @staticmethod
    def _barrowman_boattail(R_fore: float, R_aft: float, L: float,
                             x_fore: float, R_ref: float) -> tuple:
        """Barrowman CN_alpha and CP for a conical boattail or shoulder transition.

        Derivation (slender-body theory):
            Normal force per unit length ∝ d(r²)/dx for the transition.
            Integrating: CN_alpha = 2·(A_aft − A_fore) / A_ref
            For a boattail (R_aft < R_fore) this is NEGATIVE — the aft narrowing
            is destabilising (reduces the effective restoring moment).

        CP formula (centroid of the d(r²)/dx distribution for a linear frustum):
            x_CP_from_fore = L · (R_fore + 2·R_aft) / (3·(R_fore + R_aft))

        Args:
            R_fore: Forward radius of the transition (m).
            R_aft: Aft radius of the transition (m).
            L: Axial length of the transition (m).
            x_fore: Axial position of the transition's forward face from nose tip (m).
            R_ref: Reference body radius (m).

        Returns:
            (CN_alpha, x_CP) — CN_alpha in 1/rad (may be negative), x_CP in metres from nose tip.
        """
        A_ref = np.pi * R_ref ** 2
        A_fore = np.pi * R_fore ** 2
        A_aft  = np.pi * R_aft  ** 2

        CN_alpha = 2.0 * (A_aft - A_fore) / A_ref

        if abs(R_fore + R_aft) < 1e-12:
            x_cp_from_fore = L / 2.0
        else:
            x_cp_from_fore = L * (R_fore + 2.0 * R_aft) / (3.0 * (R_fore + R_aft))

        x_CP = x_fore + x_cp_from_fore

        return CN_alpha, x_CP

    # ------------------------------------------------------------------ #
    #  Barrowman aggregation                                               #
    # ------------------------------------------------------------------ #

    def compute_cnalpha_barrowman(self) -> dict:
        """Compute per-component CN_alpha and effective CP using Barrowman equations.

        Aggregates four aerodynamic components following Barrowman (1967):

            CN_alpha_total = CN_alpha_nose
                           + CN_alpha_canards   (0 if setCanardGeometry not called)
                           + CN_alpha_fins      (main fins)
                           + CN_alpha_tail      (boattail, usually negative)

            CP_effective = Σ(CN_alpha_i · CP_i) / CN_alpha_total

        After calling this method the following attributes are updated:
            self.CN_alpha_nose, self.CN_alpha_canards,
            self.CN_alpha_fins, self.CN_alpha_tail
            self.CP_nose, self.CP_canards, self.CP_fins, self.CP_tail
            self.Cnalpha_rocket  ← total, used by EOM for lift and pitching moment
            self.Cnalpha_fin     ← per-fin value for main fins, used in roll EOM

        Prerequisites:
            setRocketParams()   — provides self.d (reference diameter)
            setNoseGeometry()   — provides nose geometry
            setFinParams()      — provides main fin geometry (N, Cr, Ct, s)
            setMainFinGeometry()— provides main fin position and sweep
            setCanardGeometry() — optional, provides canard geometry
            setTailGeometry()   — optional, provides boattail geometry

        Returns:
            dict with entries 'nose', 'canards', 'fins', 'tail' each containing
            {'CN_alpha': float, 'CP': float}, plus top-level keys
            'CN_alpha_total' and 'CP_effective'.

        Raises:
            RuntimeError: If mandatory geometry (nose, main fins) is not set.
        """
        R_ref = self.d / 2.0
        results = {}

        # ---- 1. Nose cone ----
        if not hasattr(self, 'L_nose') or self.L_nose is None:
            raise RuntimeError(
                "Nose geometry not set. Call setNoseGeometry() before compute_cnalpha_barrowman()."
            )
        R_nose = (self.R_nose if (hasattr(self, 'R_nose') and self.R_nose is not None)
                  else R_ref)
        shape  = getattr(self, 'nose_shape', 'conical')
        cn_n, cp_n = self._barrowman_nose(self.L_nose, R_nose, R_ref, shape)
        self.CN_alpha_nose = cn_n
        self.CP_nose       = cp_n
        results['nose'] = {'CN_alpha': cn_n, 'CP': cp_n}

        # ---- 2. Canards ----
        if hasattr(self, 'x_canard_LE') and self.x_canard_LE is not None:
            cn_c_tot, cn_c_per, cp_c = self._barrowman_finset(
                self.N_canards, self.Cr_canards, self.Ct_canards, self.s_canards,
                self.R_body_at_canard, R_ref, self.x_canard_LE,
                getattr(self, 'x_sweep_canards', 0.0),
            )
            self.CN_alpha_canards = cn_c_tot
            self.CP_canards       = cp_c
            results['canards'] = {'CN_alpha': cn_c_tot, 'CP': cp_c}
        else:
            self.CN_alpha_canards = 0.0
            self.CP_canards       = 0.0
            results['canards'] = {'CN_alpha': 0.0, 'CP': 0.0}

        # ---- 3. Main fins ----
        if not hasattr(self, 'x_fin_LE') or self.x_fin_LE is None:
            raise RuntimeError(
                "Main fin position not set. Call setMainFinGeometry() before compute_cnalpha_barrowman()."
            )
        R_body_fin = (self.R_body_at_fin
                      if (hasattr(self, 'R_body_at_fin') and self.R_body_at_fin is not None)
                      else R_ref)
        cn_f_tot, cn_f_per, cp_f = self._barrowman_finset(
            self.N, self.Cr, self.Ct, self.s,
            R_body_fin, R_ref, self.x_fin_LE,
            getattr(self, 'x_sweep_fin', 0.0),
        )
        self.CN_alpha_fins = cn_f_tot
        self.CP_fins       = cp_f
        # Per-fin value used by the roll-moment EOM (Cnalpha_fin * N_fins in set_moments)
        self.Cnalpha_fin   = cn_f_per
        results['fins'] = {'CN_alpha': cn_f_tot, 'CP': cp_f}

        # ---- 4. Tail (boattail) ----
        if hasattr(self, 'tail_type') and self.tail_type == 'boattail':
            cn_t, cp_t = self._barrowman_boattail(
                self.R_boattail_fore, self.R_boattail_aft,
                self.L_boattail, self.x_boattail, R_ref,
            )
            self.CN_alpha_tail = cn_t
            self.CP_tail       = cp_t
            results['tail'] = {'CN_alpha': cn_t, 'CP': cp_t}
        else:
            self.CN_alpha_tail = 0.0
            self.CP_tail       = 0.0
            results['tail'] = {'CN_alpha': 0.0, 'CP': 0.0}

        # ---- Aggregate ----
        components = [
            (results['nose']['CN_alpha'],    results['nose']['CP']),
            (results['canards']['CN_alpha'], results['canards']['CP']),
            (results['fins']['CN_alpha'],    results['fins']['CP']),
            (results['tail']['CN_alpha'],    results['tail']['CP']),
        ]
        total_cn     = sum(cn for cn, _ in components)
        total_moment = sum(cn * cp for cn, cp in components)
        CP_eff = total_moment / total_cn if abs(total_cn) > 1e-9 else 0.0

        # Overwrite Cnalpha_rocket so EOM uses the Barrowman-computed total
        self.Cnalpha_rocket = total_cn

        results['CN_alpha_total'] = total_cn
        results['CP_effective']   = CP_eff

        return results



    def set_aero_direct(
        self,
        nose_cn:   float, nose_cp:   float,
        canard_cn: float, canard_cp: float,
        fin_cn:    float, fin_cp:    float,
        tail_cn:   float, tail_cp:   float,
    ):
        """Set aerodynamic coefficients directly from component-level values.

        Accepts each component's CN_alpha and CP position (all from RocketPy or
        OpenRocket) and replicates the Barrowman aggregation without needing
        any geometric inputs.

        Args:
            nose_cn   / nose_cp   : Nose cone CN_alpha (1/rad) and CP from nose (m).
            canard_cn / canard_cp : Canard set total CN_alpha and CP from nose (m).
            fin_cn    / fin_cp    : Main fin set total CN_alpha and CP from nose (m).
            tail_cn   / tail_cp   : Tail / boattail CN_alpha and CP from nose (m).
                                    Usually negative for a boattail.

        Sets:
            self.CN_alpha_nose / canards / fins / tail
            self.CP_nose / canards / fins / tail
            self.Cnalpha_rocket  — total, used by the aerodynamic EOM
            self.Cnalpha_fin     — per-fin (fin_cn / N), used by the roll-moment EOM
            self.CP_func         — constant function returning CP_effective
        """
        self.CN_alpha_nose    = float(nose_cn)
        self.CP_nose          = float(nose_cp)
        self.CN_alpha_canards = float(canard_cn)
        self.CP_canards       = float(canard_cp)
        self.CN_alpha_fins    = float(fin_cn)
        self.CP_fins          = float(fin_cp)
        self.CN_alpha_tail    = float(tail_cn)
        self.CP_tail          = float(tail_cp)

        components = [
            (nose_cn,   nose_cp),
            (canard_cn, canard_cp),
            (fin_cn,    fin_cp),
            (tail_cn,   tail_cp),
        ]
        total_cn     = sum(cn for cn, _ in components)
        total_moment = sum(cn * cp for cn, cp in components)
        cp_eff = total_moment / total_cn if abs(total_cn) > 1e-9 else 0.0

        self.Cnalpha_rocket = float(total_cn)
        # Per-fin value used by roll-moment EOM (N must already be set via setFinParams)
        if hasattr(self, 'N') and self.N:
            self.Cnalpha_fin = float(fin_cn) / float(self.N)
        else:
            self.Cnalpha_fin = float(fin_cn)  # fallback: assume N=1

        _cp = float(cp_eff)
        self.CP_func = lambda _: _cp

        _SM_launch  = (_cp - self.x_CG_0) / self.d
        _SM_burnout = (_cp - self.x_CG_f) / self.d
        print(f"\n--- Direct aero input ---")
        print(f"  {'Component':<10s}  {'CN_alpha':>10s}  {'CP (m)':>10s}")
        print(f"  {'nose':<10s}  {nose_cn:>10.4f}  {nose_cp:>10.4f}")
        print(f"  {'canards':<10s}  {canard_cn:>10.4f}  {canard_cp:>10.4f}")
        print(f"  {'fins':<10s}  {fin_cn:>10.4f}  {fin_cp:>10.4f}")
        print(f"  {'tail':<10s}  {tail_cn:>10.4f}  {tail_cp:>10.4f}")
        print(f"  {'TOTAL':<10s}  {total_cn:>10.4f}  CP_eff = {_cp:.4f} m from nose")
        print(f"  SM @ launch   (CG={self.x_CG_0:.3f} m): {_SM_launch:+.2f} cal")
        print(f"  SM @ burnout  (CG={self.x_CG_f:.3f} m): {_SM_burnout:+.2f} cal\n")

    def setEnvParams(self, v_wind: list, rho: float, g: float):
        """Set the environmental parameters.

        Args:
            v_wind (list): Wind velocity vector [x, y] in m/s.
            rho (float): Air density in kg/m^3.
            g (float): Gravitational acceleration in m/s^2.
        """
        self.v_wind = v_wind
        self.rho = rho
        self.g = g
        
        
    def setThrustCurve(self, thrust_times: np.ndarray, thrust_forces: np.ndarray):
        """Set the thrust curve data.

        Args:
            thrust_times (np.ndarray): Array of time points in seconds.
            thrust_forces (np.ndarray): Array of thrust values in Newtons.
        """
        self.thrust_times  = np.array(thrust_times,  dtype=float)
        self.thrust_forces = np.array(thrust_forces, dtype=float)

    def setThrustCurveFromFile(self, thrust_file: str):
        """Load a thrust curve from a .eng or .csv file and store it.

        For .eng files, comment lines and the metadata header are ignored and
        only 2-column ``time thrust`` rows are used.
        """
        if thrust_file.endswith(".csv"):
            df = pd.read_csv(thrust_file)
            thrust_times = df["# Time (s)"].to_numpy(dtype=float)
            thrust_forces = df["Thrust (N)"].to_numpy(dtype=float)
        elif thrust_file.endswith(".eng"):
            rows = []
            with open(thrust_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith(";"):
                        continue

                    parts = line.split()
                    if len(parts) != 2:
                        continue

                    try:
                        t_i = float(parts[0])
                        thrust_i = float(parts[1])
                    except ValueError:
                        continue

                    rows.append((t_i, thrust_i))

            if not rows:
                raise ValueError(f"No thrust samples found in {thrust_file}")

            thrust_times = np.array([row[0] for row in rows], dtype=float)
            thrust_forces = np.array([row[1] for row in rows], dtype=float)
        else:
            raise ValueError("Unsupported thrust file format. Use .eng or .csv.")

        self.setThrustCurve(thrust_times=thrust_times, thrust_forces=thrust_forces)

    def setSimParams(self, dt: float, x0: np.ndarray):
        """Set the simulation parameters. Appends initial state to states list and initial time to ts list.

        Args:
            dt (float): Time step for simulation in seconds.
            x0 (np.ndarray): Initial state vector.
        """
        self.dt = dt
        self.x0 = np.array(x0, dtype=float)

    def setSimParamsFromRailAngle(
        self,
        dt: float,
        rail_button_angular_position_deg: float,
        coordinate_system_orientation: str = "nose_to_tail",
    ):
        """Set simulation parameters using RocketPy's launch-frame convention.

        RocketPy initializes the rocket's axial spin angle from the rail-button
        angular position. For ``nose_to_tail`` coordinates this is
        ``phi_init = 360 deg - angular_position``.
        """
        if coordinate_system_orientation == "nose_to_tail":
            launch_phi_deg = 360.0 - float(rail_button_angular_position_deg)
        elif coordinate_system_orientation == "tail_to_nose":
            launch_phi_deg = float(rail_button_angular_position_deg)
        else:
            raise ValueError(
                "coordinate_system_orientation must be 'nose_to_tail' or 'tail_to_nose'"
            )

        launch_phi_rad = np.deg2rad(launch_phi_deg)
        x0 = np.array(
            [
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                np.cos(launch_phi_rad / 2.0), 0.0, 0.0, np.sin(launch_phi_rad / 2.0),
            ],
            dtype=float,
        )
        self.setSimParams(dt=dt, x0=x0)

    def cp_func_for_plots(self, aoa_deg: float, mach: float = None) -> float:
        """Compatibility wrapper for plotting helpers that expect cp(aoa, mach)."""
        if self.CP_func is None:
            raise ValueError("CP_func is not set.")
        return float(self.CP_func(aoa_deg))
        
        
    def checkParamsSet(self):
        """Check if all necessary parameters have been set.

        Raises:
            ValueError: If any parameter is not set.
        """
        required_params = [
            'I_0', 'I_f', 'I_3', 'I_3_f',
            'x_CG_0', 'x_CG_f',
            'm_0', 'm_f', 'm_p',
            'd', 'L_ne',
            't_launch_rail_clearance', 't_motor_burnout', 't_estimated_apogee',
            'thrust_times', 'thrust_forces',
            'v_wind', 'rho', 'g',
            'N', 'Cr', 'Ct', 's'
        ]
        for param in required_params:
            if not hasattr(self, param):
                raise ValueError(f"Parameter '{param}' is not set. Please set all necessary parameters before proceeding.")
            
    
