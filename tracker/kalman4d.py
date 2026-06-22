"""
kalman4d.py
───────────
A lightweight 4-state Kalman filter for bounding-box centre tracking.

State vector:  x = [cx, cy, vx, vy]^T
Measurement:   z = [cx, cy]^T
Motion model:  constant velocity  (cx' = cx + vx·dt, …)
"""

import numpy as np


class KalmanFilter4D:
    """
    Constant-velocity Kalman filter operating on the centre of a
    bounding box.

    Parameters
    ----------
    dt : float
        Time step between consecutive frames (default 1.0 = one frame).
    process_noise : float
        Scalar multiplier for the process-noise covariance Q.
    measurement_noise : float
        Scalar multiplier for the measurement-noise covariance R.
    """

    def __init__(self, dt=1.0, process_noise=1.0, measurement_noise=1.0):
        self.dt = dt

        # ── State-transition matrix  F  (4×4) ───────────
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=np.float64)

        # ── Measurement matrix  H  (2×4) ────────────────
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        # ── Process noise  Q  (4×4) ─────────────────────
        # Piecewise-constant white-noise acceleration model
        q = process_noise
        dt2 = dt ** 2
        dt3 = dt ** 3
        dt4 = dt ** 4
        self.Q = q * np.array([
            [dt4 / 4, 0,       dt3 / 2, 0      ],
            [0,       dt4 / 4, 0,       dt3 / 2],
            [dt3 / 2, 0,       dt2,     0      ],
            [0,       dt3 / 2, 0,       dt2    ],
        ], dtype=np.float64)

        # ── Measurement noise  R  (2×2) ─────────────────
        self.R = measurement_noise * np.eye(2, dtype=np.float64)

        # Placeholders (set in initiate / predict / update)
        self.x = None          # state mean  (4,)
        self.P = None          # state covariance (4,4)

    # ─────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────

    def initiate(self, measurement):
        """
        Initialise the filter from a first measurement.

        Parameters
        ----------
        measurement : array-like, shape (2,)
            First observed centre [cx, cy].

        Returns
        -------
        x : ndarray (4,)   – initial state mean
        P : ndarray (4,4)  – initial state covariance
        """
        cx, cy = measurement
        self.x = np.array([cx, cy, 0.0, 0.0], dtype=np.float64)

        # Large initial uncertainty on velocity, small on position
        self.P = np.diag([
            self.R[0, 0],      # position x
            self.R[1, 1],      # position y
            100.0,             # velocity x  (unknown)
            100.0,             # velocity y  (unknown)
        ])

        return self.x.copy(), self.P.copy()

    def predict(self):
        """
        Propagate the state one time-step forward.

        Returns
        -------
        x : ndarray (4,)  – predicted state mean
        P : ndarray (4,4) – predicted state covariance
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy(), self.P.copy()

    def update(self, measurement):
        """
        Incorporate a new measurement [cx, cy].

        Parameters
        ----------
        measurement : array-like, shape (2,)
            Observed centre [cx, cy].

        Returns
        -------
        x : ndarray (4,)  – updated state mean
        P : ndarray (4,4) – updated state covariance
        """
        z = np.asarray(measurement, dtype=np.float64)

        # Innovation
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Covariance update  (Joseph form for numerical stability)
        I_KH = np.eye(4) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        return self.x.copy(), self.P.copy()

    # ─────────────────────────────────────────────
    # Convenience helpers
    # ─────────────────────────────────────────────

    @property
    def position(self):
        """Return current estimated centre [cx, cy]."""
        return self.x[:2].copy()

    @property
    def velocity(self):
        """Return current estimated velocity [vx, vy]."""
        return self.x[2:].copy()
