"""Trading strategy simulators for forecast-driven battery arbitrage studies."""

import sys

sys.path.append("src/")

import numpy as np


def trading_quantile_strategy(
    prediction_interval,
    prediction_price_day_ahead,
    price_day_ahead,
    efficiency: float = 0.9,
):
    storage = 1
    number_days = price_day_ahead.shape[0]
    number_hours = price_day_ahead.shape[1]

    trade_list = []

    for day in range(number_days - 1):
        trade_dict = {}
        trade_dict["day"] = day
        trade_dict["hour_buy"] = []
        trade_dict["hour_sell"] = []
        trade_dict["prediction_profit"] = None
        trade_dict["make_limit_orders"] = None
        trade_dict["storage"] = None
        trade_dict["profit_order"] = []

        if storage == 1:
            hour_buy = np.argmin(prediction_price_day_ahead[day, :])
            hour_sell = np.argmax(prediction_price_day_ahead[day, :])

            sell_limit_order = prediction_interval[0, day, hour_sell]
            buy_limit_order = prediction_interval[1, day, hour_buy]
            prediction_profit = (
                sell_limit_order * efficiency - buy_limit_order / efficiency
            )
            trade_dict["prediction_profit"] = prediction_profit

            if prediction_profit > 0:
                # excecute trades
                trade_dict["make_limit_orders"] = True
                if sell_limit_order <= price_day_ahead[day, hour_sell]:
                    trade_dict["hour_sell"].append(hour_sell)
                    trade_dict["profit_order"].append(
                        price_day_ahead[day, hour_sell] * efficiency
                    )
                    storage -= 1
                if buy_limit_order >= price_day_ahead[day, hour_buy]:
                    trade_dict["hour_buy"].append(hour_buy)
                    trade_dict["profit_order"].append(
                        -price_day_ahead[day, hour_buy] / efficiency
                    )
                    storage += 1
            else:
                trade_dict["make_limit_orders"] = False

        elif storage == 0:
            hour_buy_market_order = 0
            hour_sell = 1
            hour_buy = 2

            # start point of optimization
            objective_function_max = (
                -prediction_price_day_ahead[day, hour_buy_market_order] / efficiency
                + prediction_price_day_ahead[day, hour_sell] * efficiency
                - prediction_price_day_ahead[day, hour_buy] / efficiency
            )

            # optimization
            for h1_buy in range(number_hours):
                for h2_sell in range(number_hours):
                    if h1_buy == h2_sell:
                        continue
                    for h_buy_market_order in range(number_hours):
                        if (
                            h_buy_market_order >= h2_sell
                            or h_buy_market_order == h1_buy
                        ):
                            continue

                        objective_function = (
                            -prediction_price_day_ahead[day, h_buy_market_order]
                            / efficiency
                            + prediction_price_day_ahead[day, h2_sell] * efficiency
                            - prediction_price_day_ahead[day, h1_buy] / efficiency
                        )

                        if objective_function > objective_function_max:
                            hour_buy_market_order = h_buy_market_order
                            hour_sell = h2_sell
                            hour_buy = h1_buy
                            objective_function_max = objective_function

            buy_market_order = prediction_price_day_ahead[day, hour_buy_market_order]
            sell_limit_order = prediction_interval[0, day, hour_sell]
            buy_limit_order = prediction_interval[1, day, hour_buy]

            hour_buy_min = np.argmin(prediction_price_day_ahead[day, :])
            buy_market_order_min = prediction_price_day_ahead[day, hour_buy_min]

            prediction_profit_limit = (
                -buy_market_order / efficiency
                + sell_limit_order * efficiency
                - buy_limit_order / efficiency
            )
            prediction_profit_buy_min = -buy_market_order_min / efficiency

            # excecute market and limit orders
            if prediction_profit_limit > prediction_profit_buy_min:
                trade_dict["make_limit_orders"] = True
                trade_dict["prediction_profit"] = prediction_profit_limit
                trade_dict["hour_buy"].append(hour_buy_market_order)
                trade_dict["profit_order"].append(
                    -price_day_ahead[day, hour_buy_market_order] / efficiency
                )
                storage += 1

                if sell_limit_order <= price_day_ahead[day, hour_sell]:
                    trade_dict["hour_sell"].append(hour_sell)
                    trade_dict["profit_order"].append(
                        price_day_ahead[day, hour_sell] * efficiency
                    )
                    storage -= 1

                if buy_limit_order >= price_day_ahead[day, hour_buy]:
                    trade_dict["hour_buy"].append(hour_buy)
                    trade_dict["profit_order"].append(
                        -price_day_ahead[day, hour_buy] / efficiency
                    )
                    storage += 1

            else:
                trade_dict["make_limit_orders"] = False
                trade_dict["prediction_profit"] = prediction_profit_buy_min
                trade_dict["hour_buy"].append(hour_buy_min)
                trade_dict["profit_order"].append(
                    -price_day_ahead[day, hour_buy_min] / efficiency
                )
                storage += 1

        elif storage == 2:
            hour_sell_market_order = 0
            hour_sell = 1
            hour_buy = 2

            # start point of optimization
            objective_function_max = (
                prediction_price_day_ahead[day, hour_sell_market_order] * efficiency
                + prediction_price_day_ahead[day, hour_sell] * efficiency
                - prediction_price_day_ahead[day, hour_buy] / efficiency
            )

            # optimization
            for h1_buy in range(number_hours):
                for h2_sell in range(number_hours):
                    if h1_buy == h2_sell:
                        continue
                    for h_sell_market_order in range(number_hours):
                        if (
                            h_sell_market_order >= h1_buy
                            or h_sell_market_order == h2_sell
                        ):
                            continue

                        objective_function = (
                            prediction_price_day_ahead[day, h_sell_market_order]
                            * efficiency
                            + prediction_price_day_ahead[day, h2_sell] * efficiency
                            - prediction_price_day_ahead[day, h1_buy] / efficiency
                        )

                        if objective_function > objective_function_max:
                            hour_sell_market_order = h_sell_market_order
                            hour_sell = h2_sell
                            hour_buy = h1_buy
                            objective_function_max = objective_function

            sell_market_order = prediction_price_day_ahead[day, hour_sell_market_order]
            sell_limit_order = prediction_interval[0, day, hour_sell]
            buy_limit_order = prediction_interval[1, day, hour_buy]

            hour_sell_max = np.argmax(prediction_price_day_ahead[day, :])
            sell_market_order_max = prediction_price_day_ahead[day, hour_sell_max]

            prediction_profit_limit = (
                sell_market_order * efficiency
                + sell_limit_order * efficiency
                - buy_limit_order / efficiency
            )
            prediction_profit_sell_max = sell_market_order_max * efficiency

            # excecute limit orders
            if prediction_profit_limit > prediction_profit_sell_max:
                trade_dict["make_limit_orders"] = True
                trade_dict["prediction_profit"] = prediction_profit_limit
                trade_dict["hour_sell"].append(hour_sell_market_order)
                trade_dict["profit_order"].append(
                    price_day_ahead[day, hour_sell_market_order] * efficiency
                )
                storage -= 1

                if sell_limit_order <= price_day_ahead[day, hour_sell]:
                    trade_dict["hour_sell"].append(hour_sell)
                    trade_dict["profit_order"].append(
                        price_day_ahead[day, hour_sell] * efficiency
                    )
                    storage -= 1
                if buy_limit_order >= price_day_ahead[day, hour_buy]:
                    trade_dict["hour_buy"].append(hour_buy)
                    trade_dict["profit_order"].append(
                        -price_day_ahead[day, hour_buy] / efficiency
                    )
                    storage += 1
            else:
                trade_dict["make_limit_orders"] = False
                trade_dict["prediction_profit"] = prediction_profit_sell_max
                trade_dict["hour_sell"].append(hour_sell_max)
                trade_dict["profit_order"].append(
                    price_day_ahead[day, hour_sell_max] * efficiency
                )
                storage -= 1
        else:
            print(f"Error: Storage = {storage}")
            break

        trade_dict["storage"] = storage
        trade_list.append(trade_dict)

    # last day storage should be 1
    last_day = number_days - 1

    trade_dict = {}
    trade_dict["day"] = number_days
    trade_dict["hour_buy"] = []
    trade_dict["hour_sell"] = []
    trade_dict["prediction_profit"] = 0
    trade_dict["make_limit_orders"] = None
    trade_dict["storage"] = 1
    trade_dict["profit_order"] = []
    if storage == 0:
        hour_buy = np.argmin(prediction_price_day_ahead[last_day, :])
        buy_market_order = prediction_interval[1, last_day, hour_buy]
        trade_dict["make_limit_orders"] = None
        trade_dict["prediction_profit"] = -buy_market_order / efficiency
        trade_dict["hour_buy"].append(hour_buy)
        trade_dict["profit_order"].append(
            -price_day_ahead[last_day, hour_buy] / efficiency
        )
        trade_dict["storage"] = 1
        storage += 1
        trade_list.append(trade_dict)
    elif storage == 2:
        hour_sell = np.argmax(prediction_price_day_ahead[last_day, :])
        sell_market_order = prediction_interval[0, last_day, hour_sell]
        trade_dict["make_limit_orders"] = None
        trade_dict["prediction_profit"] = sell_market_order * efficiency
        trade_dict["hour_sell"].append(hour_sell)
        trade_dict["profit_order"].append(
            price_day_ahead[last_day, hour_sell] * efficiency
        )
        trade_dict["storage"] = 1
        storage -= 1
        trade_list.append(trade_dict)
    else:
        trade_list.append(trade_dict)

    return trade_list


def trading_unlimited_bids(
    prediction_price_day_ahead,
    price_day_ahead,
    efficiency: float = 0.9,
):
    number_days = price_day_ahead.shape[0]

    trade_list = []

    for day in range(number_days - 1):
        trade_dict = {}
        trade_dict["day"] = day
        trade_dict["hour_buy"] = []
        trade_dict["hour_sell"] = []
        trade_dict["prediction_profit"] = None
        trade_dict["profit_order"] = []

        hour_buy = np.argmin(prediction_price_day_ahead[day, :])
        hour_sell = np.argmax(prediction_price_day_ahead[day, :])

        sell_order = prediction_price_day_ahead[day, hour_sell]
        buy_order = prediction_price_day_ahead[day, hour_buy]
        prediction_profit = sell_order * efficiency - buy_order / efficiency
        trade_dict["prediction_profit"] = prediction_profit

        if prediction_profit > 0:
            # excecute trades
            trade_dict["hour_sell"].append(hour_sell)
            trade_dict["profit_order"].append(
                price_day_ahead[day, hour_sell] * efficiency
            )
            trade_dict["hour_buy"].append(hour_buy)
            trade_dict["profit_order"].append(
                -price_day_ahead[day, hour_buy] / efficiency
            )
        trade_dict["storage"] = 1
        trade_list.append(trade_dict)

    return trade_list


def trading_fixed_hours(
    price_day_ahead,
    price_day_ahead_calibration,
    efficiency: float = 0.9,
):
    price_day_ahead_calibration_mean = np.mean(price_day_ahead_calibration, axis=0)
    hour_sell = np.argmax(price_day_ahead_calibration_mean)
    hour_buy = np.argmin(price_day_ahead_calibration_mean)

    trade_dict = {}
    trade_dict["hour_buy"] = hour_buy
    trade_dict["hour_sell"] = hour_sell
    trade_dict["profit_order"] = np.array(
        [
            price_day_ahead[:, hour_sell] * efficiency,
            -price_day_ahead[:, hour_buy] / efficiency,
        ]
    ).flatten()
    trade_dict["efficiency"] = efficiency

    return trade_dict


def trading_optimal_bids(
    price_day_ahead,
    efficiency: float = 0.9,
):
    hour_sell = np.argmax(price_day_ahead, axis=1)
    hour_buy = np.argmin(price_day_ahead, axis=1)

    trade_dict = {}
    trade_dict["hour_buy"] = []
    trade_dict["hour_sell"] = []
    trade_dict["profit_order"] = []

    for i in range(price_day_ahead.shape[0]):
        if (
            price_day_ahead[i, hour_sell[i]] * efficiency
            - price_day_ahead[i, hour_buy[i]] / efficiency
            > 0
        ):
            trade_dict["hour_sell"].append(hour_sell[i])
            trade_dict["hour_buy"].append(hour_buy[i])
            trade_dict["profit_order"].append(
                price_day_ahead[i, hour_sell[i]] * efficiency
            )
            trade_dict["profit_order"].append(
                -price_day_ahead[i, hour_buy[i]] / efficiency
            )
    trade_dict["efficiency"] = efficiency

    return trade_dict
