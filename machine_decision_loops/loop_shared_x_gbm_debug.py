from md_loops_debug import main_debug_loop

__author__ = "nikos.daniilidis"

if __name__ == '__main__':
    main_debug_loop(
        note='shared-x',
        readme='''n streams sharing x values, all chisq, different balances, same weights''',
        event_types=3 * ['chisq'],  # distribution of the hidden score for each stream
        seed_events=20,  # number of events to use on the first round of training
        update_events=20,  # number of total events occurring in each round of batch update
        analysis_events=20,  # number of events to use on each round of analysis
        ps=[0.47, 0.48, 0.49],  # fraction of class 1 examples in each stream
        noise_spread=0.01,  # magnitude of noise to add to the event class, as a fraction of the underlying score
        seeds=[11, 10, 9],  # random seeds for each stream
        gs=3 * [1.],  # gains to use in weighing each stream probability
        num_inputs=10,  # number of inputs in each stream
        classifier_kinds=3 * ['gbm'],  # classifier to use
        train_args={'learning_rate': [0.01, 0.03, 0.1, 0.3],
                    'n_estimators': [300],
                    'subsample': 0.5,
                    'max_depth': [3],
                    'random_state': 13,
                    'scoring': 'neg_log_loss'},
        criterion='competing_streams',  # type of selection condition
        batch_updates=10  # number of batch updates to run for the models
    )
