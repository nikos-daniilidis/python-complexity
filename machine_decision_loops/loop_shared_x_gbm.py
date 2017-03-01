from md_loops import main_loop
from config import GBM, BETA_TREE, EPS_GREEDY, COMPETING_STREAMS

__author__ = "nikos.daniilidis"

if __name__ == '__main__':
    note = 'shared-x'
    readme = '''n streams sharing x values, all chisq, different balances, same weights'''
    event_types = 10 * ['chisq']  # distribution of the hidden score for each stream
    seed_events = 10000  # number of events to use on the first round of training
    update_events = 10000  # number of total events occurring in each round of batch update
    analysis_events = 10000  # number of events to use on each round of analysis
    ps = [0.47, 0.47, 0.48, 0.49, 0.50, 0.50,  0.51, 0.52, 0.53, 0.53]  # fraction of class 1 examples in each stream
    ps = [p/5. for p in ps]  # shift class balance down to 10%
    noise_spread = 0.1  # magnitude of noise to add to the event class, as a fraction of the underlying score
    seeds = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18]  # random seeds for each stream
    gs = 10 * [1.]  # gains to use in weighing each stream probability
    num_inputs = 30  # number of inputs in each stream
    classifier_kinds = 10 * [GBM]  # classifier to use
    train_args = {'learning_rate': [0.01, 0.03, 0.06, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
                  'n_estimators': [400],
                  'subsample': 0.5,
                  'max_depth': [3],
                  'random_state': 13,
                  'scoring': 'neg_log_loss'}
    ve_config = {'kind': BETA_TREE, 'replace': True}
    criterion = COMPETING_STREAMS  # type of selection condition
    method = EPS_GREEDY
    batch_updates = 30  # number of batch updates to run for the models
    tau_dict = {ix: 0.1 for ix in range(batch_updates)}

    offsets = [0, 10, 20, 30, 40]  # I have to repeat 30 and 40 after adding the ValueError check in scorer
    for ix, seed_offset in enumerate(offsets):
        print ("\n\n---------------------")
        print("Running loop %d of %d" % (ix+1, len(offsets)))
        print ("---------------------\n\n")
        sds = [s + seed_offset for s in seeds]
        main_loop(
            note=note,
            readme=readme,
            event_types=event_types,
            seed_events=seed_events,
            update_events=update_events,
            analysis_events=analysis_events,
            ps=ps,
            noise_spread=noise_spread,
            seeds=sds,
            gs=gs,
            num_inputs=num_inputs,
            classifier_kinds=classifier_kinds,
            ve_config=ve_config,
            train_args=train_args,
            criterion=criterion,
            method=method,
            batch_updates=batch_updates,
            tau_dict=tau_dict
                )

