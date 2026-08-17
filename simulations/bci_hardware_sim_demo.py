"""
BCI Hardware Simulation Demo — 3-Layer Spherical Head Model + Motor Imagery Dipoles → 64-Channel Virtual EEG

Physical Model:
  Brain layer (r1=79mm) → Skull layer (r2=85mm) → Scalp layer (r3=92mm)
  Conductivities: sigma_brain=0.33 S/m, sigma_skull=0.0042 S/m, sigma_scalp=0.33 S/m

Forward Problem: Given cortical dipoles, analytically solve scalp potentials (3-shell spherical model)
Simulation: Motor imagery -> left/right hand C3/C4 dipole activation -> 64-channel EEG -> CSP+LDA classification
"""

import numpy as np
from scipy.special import lpmv
import time

print("=" * 60)
print("BCI Hardware Simulation — 3-Layer Spherical Head Model")
print("=" * 60)

# ============================================================
# 1. Head Model Geometry & Conductivity Parameters
# ============================================================
r_brain = 0.079    # Brain radius (m)
r_skull = 0.085    # Skull outer radius (m)
r_scalp = 0.092    # Scalp radius (m)

sigma_brain = 0.33    # Brain conductivity (S/m)
sigma_skull = 0.0042  # Skull conductivity (S/m) — low conductivity causes EEG spatial blurring
sigma_scalp = 0.33    # Scalp conductivity (S/m)

print(f"\nHead Model Parameters:")
print(f"   Brain  r={r_brain*1000:.0f}mm  sigma={sigma_brain} S/m")
print(f"   Skull  r={r_skull*1000:.0f}mm  sigma={sigma_skull} S/m  <- low conductivity blurs EEG")
print(f"   Scalp  r={r_scalp*1000:.0f}mm  sigma={sigma_scalp} S/m")

# ============================================================
# 2. 64-Channel Standard 10-20 System Electrode Positions
# ============================================================
STANDARD_64 = [
    'Fp1','Fp2','F7','F8','F1','F2','Fz','F3','F4','FC1','FC2','FC5','FC6',
    'Cz','C1','C2','C3','C4','T7','T8','CP1','CP2','CP5','CP6','Pz','P1',
    'P2','P3','P4','POz','PO3','PO4','PO7','PO8','O1','O2','Oz','AF3','AF4',
    'AF7','AF8','F5','F6','FT7','FT8','FC3','FC4','C5','C6','TP7','TP8',
    'CP3','CP4','P5','P6','PO1','PO2','PO5','PO6','Iz','Nz','A1','A2'
]

def generate_64ch_positions():
    """Generate 64-channel positions on scalp sphere (approx. 10-20 extended system)"""
    n_channels = 64
    golden_ratio = (1 + np.sqrt(5)) / 2
    positions = []
    for i in range(n_channels):
        theta = np.arccos(1 - 2*(i + 0.5)/n_channels)
        phi = 2 * np.pi * i / golden_ratio
        x = r_scalp * np.sin(theta) * np.cos(phi)
        y = r_scalp * np.sin(theta) * np.sin(phi)
        z = r_scalp * np.cos(theta)
        positions.append([x, y, z])
    return np.array(positions)

electrode_pos = generate_64ch_positions()
print(f"\nElectrode Positions: {len(electrode_pos)} channels on scalp sphere surface")

# ============================================================
# 3. 3-Layer Spherical Forward Problem (Radial Dipole, Simplified)
# ============================================================
def three_shell_forward(dipole_pos, dipole_moment, electrode_positions,
                         r1, r2, r3, s1, s2, s3, n_terms=50):
    """
    3-layer spherical head model forward problem:
    Compute scalp potentials from a cortical radial dipole.

    Based on semi-analytical solution using Legendre polynomial expansion.
    Simplified from Salu (2002) for radial dipoles.
    """
    n_elec = len(electrode_positions)
    potentials = np.zeros(n_elec)

    r_d = np.linalg.norm(dipole_pos)
    if r_d < 1e-10:
        return potentials
    d_hat = dipole_pos / r_d
    Q = np.linalg.norm(dipole_moment)

    for i in range(n_elec):
        r_e = electrode_positions[i]
        cos_theta = np.dot(d_hat, r_e) / np.linalg.norm(r_e)

        V = 0.0
        for n in range(1, n_terms + 1):
            ratio12 = (s1 - s2) / (n * s1 + (n+1) * s2)
            ratio23 = (s2 - s3) / (n * s2 + (n+1) * s3)

            r1n = (r1/r2)**(2*n+1)
            r2n = (r2/r3)**(2*n+1)

            f_n = 1.0 / (1.0 + ratio12 * r1n * (1.0 + ratio23 * r2n) + ratio23 * r2n)

            Pn = lpmv(0, n, cos_theta)
            V += (Q / (4 * np.pi * s1)) * (2*n + 1) / n * (r_d/r3)**n * f_n * Pn

        potentials[i] = V

    return potentials

print(f"\nForward Solver: 3-layer spherical analytical solution, {50}-term Legendre series")

# ============================================================
# 4. Motor Imagery Dipole Configuration
# ============================================================
# Left hand imagery: right hemisphere C4 area (motor cortex)
dipole_right = np.array([0.050, -0.050, 0.060])
# Right hand imagery: left hemisphere C3 area (motor cortex)
dipole_left = np.array([-0.050, -0.050, 0.060])

moment_radial = np.array([0.0, 0.0, 1e-8])  # 10 nA*m radial dipole moment

print(f"\nMotor Imagery Dipole Configuration:")
print(f"   Left hand -> right hemisphere C4 ({dipole_right*1000} mm)")
print(f"   Right hand -> left hemisphere C3 ({dipole_left*1000} mm)")
print(f"   Dipole moment: {np.linalg.norm(moment_radial)*1e9:.1f} nA*m")

# ============================================================
# 5. Solve Forward Problem -> Leadfield
# ============================================================
print(f"\nSolving forward problem (virtual head -> virtual electrodes)...")
t0 = time.time()

V_left = three_shell_forward(dipole_left, moment_radial, electrode_pos,
                              r_brain, r_skull, r_scalp,
                              sigma_brain, sigma_skull, sigma_scalp)
V_right = three_shell_forward(dipole_right, moment_radial, electrode_pos,
                               r_brain, r_skull, r_scalp,
                               sigma_brain, sigma_skull, sigma_scalp)

t1 = time.time()
print(f"   Solved in {t1-t0:.3f}s")
print(f"   Left hand imagery — scalp potential range: [{V_left.min()*1e6:.2f}, {V_left.max()*1e6:.2f}] uV")
print(f"   Right hand imagery — scalp potential range: [{V_right.min()*1e6:.2f}, {V_right.max()*1e6:.2f}] uV")

# ============================================================
# 6. Simulate EEG Signals (noise + temporal dynamics)
# ============================================================
sfreq = 250       # Sampling rate (Hz)
duration = 4.0    # Per-trial duration (s)
n_samples = int(sfreq * duration)
n_trials_per_class = 30
n_channels = 64

freq_mu = 11.0     # Mu rhythm center frequency (Hz)
freq_beta = 22.0   # Beta rhythm center frequency (Hz)

print(f"\nEEG Simulation Parameters:")
print(f"   Sampling rate: {sfreq} Hz, per-trial: {duration}s")
print(f"   Mu rhythm: {freq_mu} Hz, Beta rhythm: {freq_beta} Hz")
print(f"   Trials per class: {n_trials_per_class}")

np.random.seed(42)
t = np.linspace(0, duration, n_samples, endpoint=False)

X_all = []
y_all = []

for trial in range(n_trials_per_class):
    for label, V_topo in enumerate([V_left, V_right]):
        signal = np.zeros((n_channels, n_samples))

        for ch in range(n_channels):
            mu = np.sin(2*np.pi*freq_mu*t + np.random.uniform(0, 2*np.pi))
            beta = 0.5*np.sin(2*np.pi*freq_beta*t + np.random.uniform(0, 2*np.pi))

            erd_depth = V_topo[ch] / (np.max(np.abs(V_topo)) + 1e-15)
            erd_factor = 1.0 - 0.6 * np.abs(erd_depth)  # 60% suppression in active region

            signal[ch] = V_topo[ch] * 1e6 * (erd_factor * (mu + beta) +
                        0.3 * np.random.randn(n_samples))

        X_all.append(signal)
        y_all.append(label)

X_all = np.array(X_all)
y_all = np.array(y_all)

print(f"\nSimulated Dataset:")
print(f"   X shape: {X_all.shape} (trials x channels x samples)")
print(f"   Classes: {n_trials_per_class} left + {n_trials_per_class} right")

# ============================================================
# 7. CSP + LDA Classification (Standard MI-BCI Pipeline)
# ============================================================
print(f"\nCSP + LDA Classification Pipeline...")

from scipy.linalg import eigh

def extract_covariances(X, y):
    """Compute normalized covariance matrices per trial"""
    n_trials = len(y)
    covs = []
    for i in range(n_trials):
        sig = X[i]
        cov = np.cov(sig)
        cov += 1e-8 * np.eye(n_channels)
        covs.append(cov)
    return np.array(covs)

def compute_csp(covs_left, covs_right, n_components=6):
    """Compute CSP spatial filters via generalized eigenvalue problem"""
    cov_left = np.mean(covs_left, axis=0)
    cov_right = np.mean(covs_right, axis=0)

    eigenvalues, eigenvectors = eigh(cov_left, cov_right)

    idx = np.argsort(eigenvalues)
    selected = np.concatenate([idx[:n_components//2], idx[-n_components//2:]])
    W = eigenvectors[:, selected]

    return W, eigenvalues[selected]

def extract_features(X, W):
    """Extract log-variance CSP features"""
    features = []
    for i in range(len(X)):
        filtered = W.T @ X[i]
        var = np.var(filtered, axis=1)
        features.append(np.log(var + 1e-10))
    return np.array(features)

# Train/test split
train_idx = np.concatenate([np.arange(0, 20), np.arange(30, 50)])
test_idx = np.concatenate([np.arange(20, 30), np.arange(50, 60)])

covs = extract_covariances(X_all, y_all)
covs_left = covs[y_all == 0]
covs_right = covs[y_all == 1]

W_csp, lambdas = compute_csp(covs_left, covs_right, n_components=6)

print(f"   CSP eigenvalues: {lambdas.round(4)}")
print(f"   (near 0 = right-imagery discriminant, near 1 = left-imagery discriminant)")

feat_train = extract_features(X_all[train_idx], W_csp)
feat_test = extract_features(X_all[test_idx], W_csp)
y_train = y_all[train_idx]
y_test = y_all[test_idx]

# LDA classifier
class LDA:
    def fit(self, X, y):
        self.mu0 = X[y==0].mean(axis=0)
        self.mu1 = X[y==1].mean(axis=0)
        Sw = np.zeros((X.shape[1], X.shape[1]))
        for c, mu in [(0, self.mu0), (1, self.mu1)]:
            Xc = X[y==c] - mu
            Sw += Xc.T @ Xc
        self.w = np.linalg.solve(Sw, self.mu1 - self.mu0)
        self.b = -0.5 * (self.mu1 + self.mu0) @ self.w

    def predict(self, X):
        return (X @ self.w + self.b > 0).astype(int)

    def predict_proba(self, X):
        scores = X @ self.w + self.b
        return 1 / (1 + np.exp(-scores))

lda = LDA()
lda.fit(feat_train, y_train)
y_pred = lda.predict(feat_test)
acc = np.mean(y_pred == y_test)

print(f"\nClassification Results:")
print(f"   Test set: {len(y_test)} trials")
print(f"   Accuracy: {acc*100:.1f}%")
print(f"   (chance level = 50%)")

proba = lda.predict_proba(feat_test)
print(f"   Mean confidence: {np.mean(np.abs(proba - 0.5) + 0.5)*100:.1f}%")

# ============================================================
# 8. Summary
# ============================================================
print(f"\n{'='*60}")
print(f"Simulation Complete! Full Pipeline:")
print(f"   Virtual dipole (motor cortex)")
print(f"   -> 3-layer spherical forward problem")
print(f"   -> 64-channel scalp potentials (with skull attenuation)")
print(f"   -> Mu/Beta rhythm + ERD dynamic simulation")
print(f"   -> CSP spatial filtering + LDA classification")
print(f"   -> Accuracy {acc*100:.1f}%")
print(f"")
print(f"Next step: Replace spherical model with SimNIBS FEM (ICBM152 real head).")
print(f"{'='*60}")
