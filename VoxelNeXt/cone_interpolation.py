import numpy as np
from sklearn.cluster import DBSCAN

# ═══════════════════════════════════════════════════
# KEGELPARAMETER (small_cone)
# ═══════════════════════════════════════════════════
CONE_HEIGHT = 0.325   # m
CONE_RADIUS = 0.114   # m (Basisradius)

def kegel_interpolation(sparse_points, ziel_punkte=20):
    """
    Interpoliert Punkte auf Kegeloberfläche.
    Input:  sparse_points (N, 3 oder 4)
    Output: dichte_punkte (ziel_punkte, 3)
    """
    pts = sparse_points[:, :3]
    zentrum = pts.mean(axis=0)
    basis_z = pts[:, 2].min()

    n_neu = max(0, ziel_punkte - len(pts))
    neue_punkte = []

    for _ in range(n_neu):
        h      = np.random.uniform(0, CONE_HEIGHT)
        r      = CONE_RADIUS * (1 - h / CONE_HEIGHT)
        winkel = np.random.uniform(0, 2 * np.pi)
        x = zentrum[0] + r * np.cos(winkel)
        y = zentrum[1] + r * np.sin(winkel)
        z = basis_z + h
        neue_punkte.append([x, y, z])

    if neue_punkte:
        return np.vstack([pts, np.array(neue_punkte)])
    return pts


def distanz_basiertes_upsampling(punktwolke,
                                  min_dist=10.0,
                                  max_dist=15.0,
                                  min_punkte=2,
                                  max_punkte=8,
                                  ziel_punkte=20,
                                  cluster_radius=0.3):
    """
    Hauptfunktion: Findet Cone-Kandidaten bei 10-15m
    und reichert sie mit Kegel-Interpolation an.
    Input:  punktwolke (N, 4) [x, y, z, intensity]
    Output: angereicherte_punktwolke (M, 4)
    """
    distanz = np.sqrt(punktwolke[:, 0]**2 + punktwolke[:, 1]**2)

    maske_fern = (distanz >= min_dist) & (distanz <= max_dist)
    fern_punkte = punktwolke[maske_fern]
    nah_punkte  = punktwolke[~maske_fern]

    if len(fern_punkte) < min_punkte:
        return punktwolke

    # DBSCAN Clustering
    db = DBSCAN(eps=cluster_radius, min_samples=min_punkte)
    labels = db.fit_predict(fern_punkte[:, :3])

    interpolierte = []
    n_cones_gefunden = 0

    for label in set(labels):
        if label == -1:
            continue  # Rauschen überspringen
        cluster = fern_punkte[labels == label]
        if min_punkte <= len(cluster) <= max_punkte:
            dichte = kegel_interpolation(cluster, ziel_punkte)
            # Intensity für neue Punkte: Mittelwert des Clusters
            if punktwolke.shape[1] == 4:
                intensity_mean = cluster[:, 3].mean()
                intensity_col  = np.full((len(dichte) - len(cluster), 1), intensity_mean)
                orig_intensity = cluster[:, 3:4]
                dichte_mit_int = np.hstack([
                    dichte[:len(cluster)],
                    orig_intensity
                ])
                neue_mit_int = np.hstack([
                    dichte[len(cluster):],
                    intensity_col
                ])
                dichte_final = np.vstack([dichte_mit_int, neue_mit_int])
            else:
                dichte_final = dichte
            interpolierte.append(dichte_final)
            n_cones_gefunden += 1

    if interpolierte:
        neue_punkte = np.vstack(interpolierte)
        print(f"  → {n_cones_gefunden} Cone-Kandidaten gefunden, "
              f"{len(neue_punkte) - n_cones_gefunden * min_punkte} Punkte hinzugefügt")
        return np.vstack([nah_punkte, fern_punkte, neue_punkte])

    return punktwolke
