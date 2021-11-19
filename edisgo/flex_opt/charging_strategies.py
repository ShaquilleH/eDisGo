import logging
import math
import os

import numpy as np
import pandas as pd

from edisgo import EDisGo
from datetime import timedelta


COLUMNS = {
    "integrated_charging_parks_df": ["edisgo_id"],
}

logger = logging.getLogger("edisgo")


def cos_p_headcurve(p,voltage_level):
    if voltage_level == "lv":
        cos_p = (-(0.1) * p) + 1.05
        return max(min(1,cos_p),0.95)

    else:
        cos_p = (-(0.2) * p) + 1.1
        return max(min(1,cos_p),0.9)


def q_factor_from_cos_p(cos_p):
    return np.tan(np.arccos(cos_p))


def q_u_curve(v_res,
              curve_parameter={"end_upper": 1.1, "start_upper": 1.05,
                               "start_lower": 0.95, "end_lower": 0.9,
                               "max_value": 1, "dead-band_value": 0,
                               "min_value": -1}):
    curve_q_set_in_percentage = np.select(
        [(v_res > curve_parameter["end_upper"]),
         (v_res <= curve_parameter["end_upper"]) &
         (v_res >= curve_parameter["start_upper"]),
         (v_res < curve_parameter["start_upper"]) &
         (v_res >= curve_parameter["start_lower"]),
         (v_res < curve_parameter["start_lower"]) &
         (v_res >= curve_parameter["end_lower"]),
         (v_res < curve_parameter["end_lower"])],
        [curve_parameter["max_value"],
         curve_parameter["max_value"] - curve_parameter["max_value"] /
         (curve_parameter["start_upper"] - curve_parameter["end_upper"]) *
         (v_res - curve_parameter["end_upper"]),
         curve_parameter["dead-band_value"],
         curve_parameter["min_value"] *
         (v_res - curve_parameter["start_lower"]) /
         (curve_parameter["end_lower"] - curve_parameter["start_lower"]),
         curve_parameter["min_value"]])

THRESHOLD = 5



def integrate_charging_parks(edisgo_obj, comp_type="ChargingPoint"):
    """
    Integrates all designated charging parks into the grid. The charging demand is
    not integrated here, but an empty dummy timeseries is generated.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    comp_type : str
        Component Type. Default "ChargingPoint"

    """
    charging_parks = list(edisgo_obj.electromobility.potential_charging_parks)

    # Only integrate charging parks with designated charging points
    designated_charging_parks = [
        cp for cp in charging_parks if (cp.designated_charging_point_capacity > 0) and cp.within_grid]

    charging_park_ids = [_.id for _ in designated_charging_parks]

    dummy_timeseries = pd.Series(
        [0] * len(edisgo_obj.timeseries.timeindex), index=edisgo_obj.timeseries.timeindex)

    # integrate ChargingPoints and save the names of the eDisGo ID
    edisgo_ids = [
        EDisGo.integrate_component(
            edisgo_obj,
            comp_type=comp_type,
            geolocation=cp.geometry,
            use_case=cp.use_case,
            add_ts=True,
            ts_active_power=dummy_timeseries,
            ts_reactive_power=dummy_timeseries,
            p_nom=cp.grid_connection_capacity,
        ) for cp in designated_charging_parks
    ]

    edisgo_obj.electromobility.integrated_charging_parks_df = pd.DataFrame(
        columns=COLUMNS["integrated_charging_parks_df"], data=edisgo_ids, index=charging_park_ids)

# TODO: the dummy timeseries should be as long as the simulated days and not the timeindex of the edisgo object
# At the moment this would result into wrong results if the timeindex of the edisgo object is
# not continuously (e.g. 2 weeks of the year)
def charging_strategy(
        edisgo_obj, strategy="dumb", timestamp_share_threshold=0.2, minimum_charging_capacity_factor=0.1,
        reactive_power_strategy=None, max_iterations=2):
    """
    Calculates the timeseries per charging park if parking times are given.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    strategy : str
        The charging strategy. Default "dumb". Only "private" charging processes at "home"
        or at "work" can be flexibilized. "Public" charging processes will always be "dumb".
        For now the following charging strategies are valid:
            "dumb": The cars are charged directly after arrival with the maximum possible
            charging capacity.
            "reduced": The cars are charged directly after arrival with the minimum possible
            charging capacity. The minimum possible charging capacity is determined by the
            parking time and the minimum_charging_capacity_factor.
            "residual": The cars are charged when the residual load in the MV grid is at it's
            lowest (high generation and low consumption). Charging processes with a low
            flexibility band are given priority.
    timestamp_share_threshold : float
        Percental threshold of the time required at a time step for charging the vehicle.
        If the time requirement is below this limit, then the charging process is not mapped
        into the time series. If, however, it is above this limit, the time step is mapped
        to 100% into the time series. This prevents differences between the charging strategies
        and creates a compromise between the simultaneity of charging processes and an
        artificial increase in the charging demand. Default 0.2
    minimum_charging_capacity_factor : float
        Technical percental minimum charging capacity per charging point. Default 0.1

    """
    def harmonize_charging_processes_df(
            df, len_ts, timestamp_share_threshold, strategy=None, minimum_charging_capacity_factor=0.1, eta_cp=1.0):
        """
        Harmonizes the charging processes to prevent differences in the energy
        demand per charging strategy.

        Parameters
        ----------
        df : :pandas:`pandas.DataFrame<DataFrame>`
            Charging processes DataFrame
        len_ts : int
            Length of the timeseries
        timestamp_share_threshold : float
            See description in the main function. Default 0.2
        strategy : str
            See description in the main function. Default "dumb"
        minimum_charging_capacity_factor : float
            See description in the main function. Default 0.1
        eta_cp : float
            Charging Point efficiency. Default 1.0

        """
        # FIXME: SimBEV has a MATLAB legacy and at the moment 1 is eDisGos 0
        df = df.assign(park_start=df.park_start-1)

        # FIXME: This should become obsolete in the future when SimBEV is bugfixed
        # drop rows that have a park start higher than simulated days
        df = df.loc[df.park_start <= len_ts]

        # calculate the minimum time taken the fulfill the charging demand
        minimum_charging_time = df.chargingdemand/df.netto_charging_capacity \
                                * 60/edisgo_obj.electromobility.stepsize

        # calculate in which time steps the last time step needed to fulfill the
        # charging demand is considered in the timeseries
        mask = (minimum_charging_time % 1) >= timestamp_share_threshold

        minimum_charging_time.loc[mask] = minimum_charging_time.apply(np.ceil)

        minimum_charging_time.loc[~mask] = minimum_charging_time.apply(np.floor)

        # recalculate the charging demand from the charging capacity
        # and the minimum charging time
        # Calculate the grid sided charging capacity in MVA
        df = df.assign(
            minimum_charging_time=minimum_charging_time.astype(np.uint16),
            harmonized_chargingdemand=minimum_charging_time * df.netto_charging_capacity * edisgo_obj.electromobility.stepsize / 60,
            netto_charging_capacity_mva=df.netto_charging_capacity.divide(10**3 * eta_cp),  # kW --> MW
        )

        if strategy == "reduced":
            parking_time = df.park_end - df.park_start

            # calculate the maximum needed charging time with the minimum charging capacity
            maximum_needed_charging_time = df.harmonized_chargingdemand /\
                                           (minimum_charging_capacity_factor * df.netto_charging_capacity)\
                                           * 60/edisgo_obj.electromobility.stepsize

            maximum_needed_charging_time = maximum_needed_charging_time.apply(np.floor).astype(np.uint16)

            # when the parking time is less than the maximum needed charging time, the total charging time
            # equates the parking time and the charging capacity is recalculated accordingly
            mask = parking_time <= maximum_needed_charging_time

            df = df.assign(
                reduced_charging_time=0,
                reduced_charging_capacity=0,
            )

            df.loc[mask, "reduced_charging_time"] = parking_time.loc[mask]

            df.loc[~mask, "reduced_charging_time"] = maximum_needed_charging_time.loc[~mask]

            df.reduced_charging_capacity = df.harmonized_chargingdemand / df.reduced_charging_time \
                                           * 60/edisgo_obj.electromobility.stepsize

            df = df.assign(
                reduced_charging_capacity_mva=df.reduced_charging_capacity.divide(10**3 * eta_cp),
            )

        elif strategy == "residual":
            # the flex time/band is defined as the amount of time steps not needed to fulfill
            # the charging demand in a parking process
            parking_time = df.park_end - df.park_start

            df = df.assign(
                flex_time=parking_time - df.minimum_charging_time
            )

            df = df.sort_values(by=["flex_time", "park_start"], ascending=[True, True])

        return df

    def _overwrite_active_power_timeseries(edisgo_obj, edisgo_id, ts):
        """
        Overwrites the dummy timeseries for the Charging Point

        Parameters
        ----------
        edisgo_obj : :class:`~.EDisGo`
        edisgo_id : str
            eDisGo ID of the Charging Point
        ts : :pandas:`pandas.Series<Series>`
            New timeseries

        """
        edisgo_obj.timeseries._charging_points_active_power.loc[:, edisgo_id] = \
            ts.loc[edisgo_obj.timeseries.timeindex]

    def _overwrite_reactive_power_timeseries(edisgo_obj, edisgo_id, ts):

        edisgo_obj.timeseries._charging_points_reactive_power.loc[:, edisgo_id] = \
            ts.loc[edisgo_obj.timeseries.timeindex]

    # get integrated charging parks
    charging_parks = [
        cp for cp in list(edisgo_obj.electromobility.potential_charging_parks)
        if cp.grid is not None
    ]

    # Reset possible old timeseries as these influence "residual" charging
    ts = pd.Series(data=0, index=edisgo_obj.timeseries.timeindex)

    for cp in charging_parks:
        _overwrite_active_power_timeseries(edisgo_obj, cp.edisgo_id, ts)

    eta_cp = edisgo_obj.electromobility.eta_charging_points

    len_ts = int(edisgo_obj.electromobility.simulated_days * 24 \
                 * 60 / edisgo_obj.electromobility.stepsize)

    timeindex = pd.date_range(
        edisgo_obj.timeseries.timeindex[0], periods=len_ts, freq=f"{edisgo_obj.electromobility.stepsize}min")

    edisgo_timedelta = edisgo_obj.timeseries.timeindex[1] - edisgo_obj.timeseries.timeindex[0]
    simbev_timedelta = timeindex[1] - timeindex[0]

    assert edisgo_timedelta == simbev_timedelta, (
        "The stepsize of the timeseries of the edisgo object differs from the simbev stepsize. " 
        f"The edisgo timedelta is {edisgo_timedelta}, while the simbev timedelta is {simbev_timedelta}. " 
        "Make sure to use a matching stepsize.")


    if strategy == "dumb":
        # "dumb" charging
        for cp in charging_parks:
            dummy_ts = np.zeros(len_ts)

            charging_processes_df = harmonize_charging_processes_df(
                cp.charging_processes_df, len_ts, timestamp_share_threshold, strategy=strategy, eta_cp=eta_cp)

            if reactive_power_strategy == "cos_P":
                reactive_power_dummy_ts = np.zeros(len_ts)

                for _, row in charging_processes_df.iterrows():
                    dummy_ts[row["park_start"]:row["park_start"] + row["minimum_charging_time"]] += \
                        row["netto_charging_capacity_mva"]

                    cos_p = cos_p_headcurve(row["netto_charging_capacity"],cp.voltage_level)

                    q_faktor = q_factor_from_cos_p(cos_p)

                    reactive_power_dummy_ts[row["park_start"]:row["park_start"] + row["minimum_charging_time"]] -= \
                        (row["netto_charging_capacity_mva"] * q_faktor)

                _overwrite_reactive_power_timeseries(
                    edisgo_obj, cp.edisgo_id, pd.Series(data=reactive_power_dummy_ts, index=timeindex))

            else:
                for _, row in charging_processes_df.iterrows():
                    dummy_ts[row["park_start"]:row["park_start"] + row["minimum_charging_time"]] += \
                        row["netto_charging_capacity_mva"]

            _overwrite_active_power_timeseries(
                edisgo_obj, cp.edisgo_id, pd.Series(data=dummy_ts, index=timeindex))

    #Fixed_cos auch hier reinnehmen?
    #        elif reactive_power_strategy == "fixed_cos":
    #            def q_factor_from_cos_p(cos_p):
    #                return math.tan(math.acos(cos_p))
    #
    #            edisgo_obj.timeseries._charging_points_reactive_power = \
    #                edisgo_obj.timeseries.charging_points_active_power * q_factor_from_cos_p(0.9) * (-1)


    elif strategy == "reduced":
        # "reduced" charging
        for cp in charging_parks:
            dummy_ts = np.zeros(len_ts)

            charging_processes_df = harmonize_charging_processes_df(
                cp.charging_processes_df, len_ts, timestamp_share_threshold, strategy=strategy,
                minimum_charging_capacity_factor=minimum_charging_capacity_factor, eta_cp=eta_cp)

            if reactive_power_strategy == "cos_P":
                reactive_power_dummy_ts = np.zeros(len_ts)
                #An der Stelle wieder Kennlinie cos_P

                for _, row in charging_processes_df.iterrows():
                    if row["use_case"] == "public":
                        # if the charging process takes place in a "public" setting
                        # the charging is "dumb"
                        dummy_ts[row["park_start"]:row["park_start"] + row["minimum_charging_time"]] += \
                            row["netto_charging_capacity_mva"]
                    else:
                        dummy_ts[row["park_start"]:row["park_start"] + row["reduced_charging_time"]] += \
                            row["reduced_charging_capacity_mva"]

            _overwrite_active_power_timeseries(
                edisgo_obj, cp.edisgo_id, pd.Series(data=dummy_ts, index=timeindex))

    elif strategy == "residual":
        # "residual" charging
        # only use charging processes from integrated charging parks
        charging_processes_df = edisgo_obj.electromobility.charging_processes_df[
            edisgo_obj.electromobility.charging_processes_df.charging_park_id.isin(
                edisgo_obj.electromobility.integrated_charging_parks_df.index)]

        charging_processes_df = harmonize_charging_processes_df(
                charging_processes_df, len_ts, timestamp_share_threshold, strategy=strategy, eta_cp=eta_cp)

        # get residual load
        init_residual_load = edisgo_obj.timeseries.residual_load

        len_residual_load = int(charging_processes_df.park_end.max())

        if len(init_residual_load) >= len_residual_load:
            init_residual_load = init_residual_load.loc[timeindex]
        else:
            while len(init_residual_load) < len_residual_load:
                len_rl = len(init_residual_load)
                len_append = min(len_rl, len_residual_load-len_rl)

                s_append = init_residual_load.iloc[:len_append]

                init_residual_load = init_residual_load.append(
                    s_append, ignore_index=True)

        init_residual_load = init_residual_load.to_numpy()

        timeindex_residual = pd.date_range(
            edisgo_obj.timeseries.timeindex[0], periods=len(init_residual_load),
            freq=f"{edisgo_obj.electromobility.stepsize}min")

        dummy_ts = pd.DataFrame(
            data=0., columns=[_.id for _ in charging_parks], index=timeindex_residual)

        # determine which charging processes can be flexibilized
        dumb_charging_processes_df = charging_processes_df.loc[
            (charging_processes_df.use_case == "public") |
            (charging_processes_df.flex_time == 0)
        ]

        flex_charging_processes_df = charging_processes_df.loc[
            ~charging_processes_df.index.isin(dumb_charging_processes_df.index)]

        # perform dumb charging processes and respect them in the residual load
        for _, row in dumb_charging_processes_df.iterrows():
            try:
                dummy_ts.loc[:, row["charging_park_id"]].iloc[
                    row["park_start"]:row["park_start"] + row["minimum_charging_time"]
                ] += row["netto_charging_capacity_mva"]
            except:
                park_start = row["park_start"]
                park_end = row["park_start"] + row["minimum_charging_time"]
                maximum_ts = len(dummy_ts)
                logger.warning(
                    (f"Charging process with index {_} could not be respected. "
                     f"The park start is at timestep {park_start} and the park end is "
                     f"at timestep {park_end}, while the timeseries consists of {maximum_ts}"
                     "timesteps."))

        residual_load = init_residual_load + dummy_ts.sum(
            axis=1).to_numpy()

        for _, row in flex_charging_processes_df.iterrows():
            flex_band = residual_load[row["park_start"]:row["park_end"]]

            k = row["minimum_charging_time"]

            # get k time steps with the lowest residual load in the parking time
            idx = np.argpartition(flex_band, k)[:k] + row["park_start"]

            try:
                dummy_ts[row["charging_park_id"]].iloc[idx] += \
                    row["netto_charging_capacity_mva"]

                residual_load[idx] += row["netto_charging_capacity_mva"]

            except:
                logger.warning(
                    (f"Charging process with index {_} could not be respected. "
                     f"The charging takes place within the timesteps {idx}, "
                     f"while the timeseries consists of {maximum_ts} timesteps."))

        for count, col in enumerate(dummy_ts.columns):
            _overwrite_active_power_timeseries(
                edisgo_obj, charging_parks[count].edisgo_id, dummy_ts[col])

    else:
        raise ValueError(
            f"Strategy {strategy} has not yet been implemented.")

    logging.info(f"Charging strategy {strategy} completed.")

    if reactive_power_strategy == "fixed_cos":
        edisgo_obj.timeseries._charging_points_reactive_power = \
            edisgo_obj.timeseries.charging_points_active_power * q_factor_from_cos_p(0.9) * (-1)


    if reactive_power_strategy == "Q_U":
        charging_points_active_power = edisgo_obj.timeseries.charging_points_active_power
        buses = edisgo_obj.topology.charging_points_df.loc[
            charging_points_active_power.columns].bus
        edisgo_obj.analyze(
            use_seed=True)
        i = 0

        while i < max_iterations:
            v_res = edisgo_obj.results.v_res[buses] - 1
            p_nom_utilisation = edisgo_obj.topology.charging_points_df.loc[
                charging_points_active_power.columns, "p_nom"].astype("float")




        #Denke es gibt bei der Q(U)-Regelung keine Unterscheidung in MV,LV
        #nur bei der cos(phi) Regelung gibt es die

        # if p_nom_utilisation < 0.3:
        #     curve_q_set_in_percentage*0.95
        #
        #
        # elif p_nom_utilisation >= 0.3:
        #     curve_q_set_in_percentage*0.9

            edisgo_obj.analyze(
            use_seed=True)

            if (v_res.round(THRESHOLD) == edisgo_obj.results.v_res[buses].round(THRESHOLD)).all().all():
                logger.info(
                    f"Stabilized Q(U) control after {i+1} iterations.")
                break

            else:
                i += 1
                logger.debug(
                    f"Finished Q(U) control iteration {i}.")

                if i == MAX_ITERATIONS:
                    logger.info(
                        "Halted Q(U) control after the maximum "
                        f"allowed iterations of {i}.")



        if reactive_power_strategy == "cos_P":
            # def cos_p_headcurve(x):
            #     return (-(0.2) * x) + 1.1
            #
            # def q_factor_from_cos_p(cos_p):
            #     return np.tan(np.arccos(cos_p))

            charging_points_active_power = edisgo_obj.timeseries.charging_points_active_power

            p_nom_utilisation2 = charging_points_active_power / edisgo_obj.topology.charging_points_df.loc[
                charging_points_active_power.columns, "p_nom"].astype("float")

            cos_p = p_nom_utilisation2.apply(cos_p_headcurve).clip(
                upper=1., lower=0.9)  #

            q_factor = cos_p.apply(q_factor_from_cos_p)

            edisgo_obj.timeseries._charging_points_reactive_power = \
                charging_points_active_power * q_factor * (-1)

            print("break")

        print("break")

            # Q_faktor = relevant_v_res * 0.33 / 0.04
            # Q_faktor = Q_faktor.clip(upper=0.33, lower=-0.33)
            # edisgo_obj.timeseries._charging_points_reactive_power = \
            #     charging_points_active_power * Q_faktor.to_numpy()
            #
            # i+= 1

    # Q(U)-Kennlinie Fredo:
    # def q_u_curve(v_res,
    #               curve_parameter={"end_upper": 1.1, "start_upper": 1.05,
    #                                "start_lower": 0.95, "end_lower": 0.9,
    #                                "max_value": 1, "dead-band_value": 0,
    #                                "min_value": -1}):
    #
    # curve_q_set_in_percentage = np.select(
    #     [(v_res > curve_parameter["end_upper"]),
    #      (v_res <= curve_parameter["end_upper"]) &
    #      (v_res >= curve_parameter["start_upper"]),
    #      (v_res < curve_parameter["start_upper"]) &
    #      (v_res >= curve_parameter["start_lower"]),
    #      (v_res < curve_parameter["start_lower"]) &
    #      (v_res >= curve_parameter["end_lower"]),
    #      (v_res < curve_parameter["end_lower"])],
    #     [curve_parameter["max_value"],
    #      curve_parameter["max_value"] - curve_parameter["max_value"] /
    #      (curve_parameter["start_upper"] - curve_parameter["end_upper"]) *
    #      (v_res - curve_parameter["end_upper"]),
    #      curve_parameter["dead-band_value"],
    #      curve_parameter["min_value"] *
    #      (v_res - curve_parameter["start_lower"]) /
    #      (curve_parameter["end_lower"] - curve_parameter["start_lower"]),
    #      curve_parameter["min_value"]])

    #Als nächstes erneut .analyze ausführen und Differenz zwischen beiden ermitteln


   # if reactive_power_strategy = "Q_U":
   #     edisgo_obj.results.v_res.analyze()

#Zwischenpunkt cos_P

        # edisgo_obj.timeseries._charging_points_reactive_power =

   # elif reactive_power_strategy == "Q_U":
    #    relevant_v_res = edisgo_obj.results.v_res[edisgo_obj.topology.charging_points_df.bus]
     #   relevant_v_res = relevant_v_res - 1
      #  Q_faktor = relevant_v_res * 0.33 / 0.04
       # Q_faktor = Q_faktor.clip(upper=0.33, lower=-0.33)

    #elif reactive_power_strategy == "Q_U":
    #    def Q_U_headcurve(x):
    #        return (8.25*x)-8.25
     #   Q_U = edisgo_obj.timeseries.charging_points_reactive_power #TODO richtigen Pfad einsetzen
    #    mask = (Q_U / edisgo_obj.topology.charging_points_df.loc[cos_P.columns,"p_nom"]) <= 0.5 größer -0.33, kleiner 0.33
    #    Q_U[mask] = 0
     #   Q_U=Q_U.apply(Q_U_headcurve)

#    elif ractive_power_strategy == "Q_U":
 #       if edisgo_obj.timeseries._charging_points_reactive_power / edisgo_obj.topology.charging_points_df.bus = 0.33
  #
   #
    #    elif
     #Für alle drei Stufen der Kennlinie einzeln schreiben




