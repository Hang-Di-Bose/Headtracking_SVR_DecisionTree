import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf 

class Seq2SeqDataset(object):
    def __init__(self,
                input_path,
                input_window_length_samples=50,
                output_window_length_samples=3,
                num_signals=4,
                # transformL=[],
                percent_val=0.2,
                percent_test=0.2,
                buffer_size=100000,
                batch_size=32):

        self.input_path = input_path 

        self.transformL = [
            ['signal_selection',    { 'sigL':['rot_x','rot_y','rot_z','rot_w'] } ] ,
            ['boseAR_scaling',      { } ],
            ['zscore_normalize',    { 'sigL':['rot_x','rot_y','rot_z','rot_w'], 'relFl':True, 'medianFl':False, 'result_label':'rot_zscore_normalize' }],
        ]
        
        self.input_window_length_samples  = input_window_length_samples
        self.output_window_length_samples = output_window_length_samples
        self.num_signals = 4

        self.buffer_size  = buffer_size
        self.batch_size   = batch_size
        self.percent_val  = percent_val
        self.percent_test = percent_test

        self._make_dataset()

        pass

    def _make_dataset(self):
        # filepaths = 
        rec_sessions = self._read_recording_sessions()
        rec_sessions_and_transformLs = [ self._apply_transformL(rs) for rs in rec_sessions ]
        rec_sessions = self._calculate_normalization_constants(rec_sessions_and_transformLs)
        rec_sessions = [ self._apply_scenario_selection(rs) for rs in rec_sessions ]
        # import ipdb; ipdb.set_trace()
        windows = self._generate_windows(rec_sessions)
        train_windowL, test_windowL, val_windowL = self._apply_train_test_val_split(windows)
        self.train_dataset = self._make_dataset_from_windowL(train_windowL, batch_size=self.batch_size, repeat=-1)
        self.test_dataset  = self._make_dataset_from_windowL(test_windowL,  batch_size=1,               repeat=-1)
        self.val_dataset   = self._make_dataset_from_windowL(val_windowL,   batch_size=1,               repeat=0)
        pass

    def _read_recording_sessions(self):
        from RecordingSession.RecordingSession import RecordingSession
        sessions = []

        filepath = self.input_path
        files = [f for f in os.listdir(filepath) if f.endswith('json')]

        for file in files:
            rs = RecordingSession(os.path.join(filepath,file))
            print('converting',rs.ctrlD['userId'],'to RecordingSession')
            sessions.append(rs)

        return sessions
    
    def _apply_transformL(self, session):
        print('apply transformL! session',session.ctrlD['userId'])
        applyTransforms_return = session.applyTransforms( self.transformL )
        print(applyTransforms_return)

        return (session, applyTransforms_return)

    def _calculate_normalization_constants( self, rec_sessions_and_transformLs ):
        # calculate cumulative normalization offset and divisor
        # scaleD_cumulative = { sig:{ 'offset':0.0, 'divisor':0.0 } for sig in self.sigL }

        rec_sessions = []

        # for (rec_session, transformL_return) in rec_sessions_and_transformLs:
        for (rec_session, _) in rec_sessions_and_transformLs:
            rec_sessions.append(rec_session)
            # for key in transformL_return.keys():
            #     if 'normalize' in key:
            #         normalize_return = transformL_return[key]
                    # for sig in normalize_return.keys():
                    #     scaleD_cumulative[sig]['offset']  += normalize_return[sig]['offset']
                    #     scaleD_cumulative[sig]['divisor'] += normalize_return[sig]['divisor']
        
        # offsetL  = []
        # divisorL = []
        # for sig in self.sigL:
        #     scaleD_cumulative[sig]['offset']  /= len(rec_sessions_and_transformLs)
        #     scaleD_cumulative[sig]['divisor'] /= len(rec_sessions_and_transformLs)

        #     offsetL.append(float(scaleD_cumulative[sig]['offset']))
        #     divisorL.append(float(scaleD_cumulative[sig]['divisor']) if scaleD_cumulative[sig]['divisor'] != 0 else 1.0)

        # self.offsetL  = offsetL
        # self.divisorL = divisorL

        return rec_sessions

    def _apply_scenario_selection(self,session):
        print('apply scenario selection! session',session.ctrlD['userId'])
        session.applyTransforms([['scenario_selection',{'scenarioL':['standing_on_face']}]])
        return session 

    def _generate_windows(self, sessions):
        def _generate_session_windows(session):
            print('generate windows! session',session.ctrlD['userId'])
            sr = session.trace_srate
            scenario_boundaryL = []
            scenario_start_event_index = 0

            for i in range(1,len(session.eventDf)):
                prev_event = session.eventDf.iloc[i-1]
                event = session.eventDf.iloc[i]

                if prev_event['index']+1 != event['index']:
                    scenario_end_event_index = i-1
                    # sanity check that it's more than 1 event long (aka, not just a stop event after a walking scenario)
                    if scenario_start_event_index != i-1:
                        scenario_boundaryL.append(
                            {
                                'start':session.eventDf.iloc[scenario_start_event_index]['start'],
                                'end':  session.eventDf.iloc[scenario_end_event_index]['end']
                            }
                        )
                    scenario_start_event_index = i

            windows = []
            for scenario_boundary in scenario_boundaryL:
                start_ms = scenario_boundary['start']
                end_ms   = scenario_boundary['end']
                start_sample = int(np.floor(start_ms / 1000 * sr))
                end_sample   = int(np.floor(end_ms   / 1000 * sr))

                # i should hop entirely over the input window and the output window
                i = start_sample
                # hop = self.input_window_length_samples + self.output_window_length_samples
                while i < end_sample - (self.input_window_length_samples + self.output_window_length_samples):
                    window = np.copy(session.trace_yM[i:i+self.input_window_length_samples,:])
                    label  = np.copy(session.trace_yM[i+self.input_window_length_samples:i+self.input_window_length_samples+self.output_window_length_samples,:])
                    print('window.shape =',window.shape,'\tlabel.shape =',label.shape)
                    # window = np.squeeze(window.reshape((-1))) # feed a 1D vector
                    windows.append((window,label))
                    i += self.input_window_length_samples + self.output_window_length_samples

            return windows 

        windows = [_generate_session_windows(session) for session in sessions]
        windows = [window for dataset in windows for window in dataset] # flatten array of arrays into single array
        print('total number of windows =',len(windows))
        return windows

    def _apply_train_test_val_split(self, windowL):
        trainL = []; num_train = 0
        testL  = []; num_test  = 0
        valL   = []; num_val   = 0

        # use random number generator to partition windows into train, test, and val groups
        for window in windowL:
            rand = random.random()
            if rand < self.percent_test:
                testL.append(window)
                num_test += 1
            elif rand < self.percent_test + self.percent_val:
                valL.append(window)
                num_val += 1
            else: # rand >= self.percent_test + self.percent_val
                trainL.append(window)
                num_train += 1

        self.num_val   = num_val
        self.num_test  = num_test
        self.num_train = num_train 

        return trainL, testL, valL

    def _make_dataset_from_windowL( self, windowL, batch_size, repeat ):
        # (signal, gesture_label, label_mid, label_return, row_start_offset, row_end_offset, mid_end_offset, return_end_offset)
        dataset = tf.data.Dataset.from_generator(lambda: windowL, 
                                                output_types=(tf.dtypes.as_dtype(np.float32), tf.dtypes.as_dtype(np.float32)),
                                                output_shapes=(tf.TensorShape(( self.input_window_length_samples, self.num_signals)), tf.TensorShape((self.output_window_length_samples,self.num_signals)) ) 
                                                ) 
        dataset = dataset.shuffle(buffer_size=self.buffer_size)
        dataset = dataset.map(self._do_flatten)
        # dataset = dataset.map(self._do_windowing)
        # dataset = dataset.map(self._do_tiling)
        # dataset = dataset.map(self._labels_to_onehot)
        ## we unbatch because not all of the gestures have the same number of windows and we want to be able 
        # to split gesture over multiple batches
        # dataset = dataset.apply(tf.data.experimental.unbatch())
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat(repeat)
        return dataset

    def _do_flatten(self,sig_in,sig_out):
        return (tf.reshape(sig_in,[-1]),tf.squeeze(sig_out))

class Seq2SeqDataset_copy(object):
    def __init__(self,
                input_path,
                input_window_length_samples=50,
                output_window_length_samples=3,
                num_signals=4,
                # transformL=[],
                percent_val=0.2,
                percent_test=0.2,
                buffer_size=100000,
                batch_size=32):

        self.input_path = input_path 

        self.transformL = [
            ['signal_selection',    { 'sigL':['rot_x','rot_y','rot_z','rot_w'] } ] ,
            ['boseAR_scaling',      { } ],
            ['zscore_normalize',    { 'sigL':['rot_x','rot_y','rot_z','rot_w'], 'relFl':True, 'medianFl':False, 'result_label':'rot_zscore_normalize' }],
        ]
        
        self.input_window_length_samples  = input_window_length_samples
        self.output_window_length_samples = output_window_length_samples
        self.num_signals = 4

        self.buffer_size  = buffer_size
        self.batch_size   = batch_size
        self.percent_val  = percent_val
        self.percent_test = percent_test

        self._make_dataset()

        pass

    def _make_dataset(self):
        # filepaths = 
        rec_sessions = self._read_recording_sessions()
        rec_sessions_and_transformLs = [ self._apply_transformL(rs) for rs in rec_sessions ]
        rec_sessions = self._calculate_normalization_constants(rec_sessions_and_transformLs)
        rec_sessions = [ self._apply_scenario_selection(rs) for rs in rec_sessions ]
        # import ipdb; ipdb.set_trace()
        windows = self._generate_scenarios(rec_sessions)
        train_windowL, test_windowL, val_windowL = self._apply_train_test_val_split(windows)
        self.train_dataset = self._make_dataset_from_windowL(train_windowL, batch_size=self.batch_size, repeat=-1)
        self.test_dataset  = self._make_dataset_from_windowL(test_windowL,  batch_size=1,               repeat=-1)
        self.val_dataset   = self._make_dataset_from_windowL(val_windowL,   batch_size=1,               repeat=0)
        pass

    def _read_recording_sessions(self):
        from RecordingSession.RecordingSession import RecordingSession
        sessions = []

        filepath = self.input_path
        files = [f for f in os.listdir(filepath) if f.endswith('json')]

        for file in files:
            rs = RecordingSession(os.path.join(filepath,file))
            print('converting',rs.ctrlD['userId'],'to RecordingSession')
            sessions.append(rs)

        return sessions
    
    def _apply_transformL(self, session):
        print('apply transformL! session',session.ctrlD['userId'])
        applyTransforms_return = session.applyTransforms( self.transformL )
        print(applyTransforms_return)

        return (session, applyTransforms_return)

    def _calculate_normalization_constants( self, rec_sessions_and_transformLs ):
        # calculate cumulative normalization offset and divisor
        # scaleD_cumulative = { sig:{ 'offset':0.0, 'divisor':0.0 } for sig in self.sigL }

        rec_sessions = []

        # for (rec_session, transformL_return) in rec_sessions_and_transformLs:
        for (rec_session, _) in rec_sessions_and_transformLs:
            rec_sessions.append(rec_session)
            # for key in transformL_return.keys():
            #     if 'normalize' in key:
            #         normalize_return = transformL_return[key]
                    # for sig in normalize_return.keys():
                    #     scaleD_cumulative[sig]['offset']  += normalize_return[sig]['offset']
                    #     scaleD_cumulative[sig]['divisor'] += normalize_return[sig]['divisor']
        
        # offsetL  = []
        # divisorL = []
        # for sig in self.sigL:
        #     scaleD_cumulative[sig]['offset']  /= len(rec_sessions_and_transformLs)
        #     scaleD_cumulative[sig]['divisor'] /= len(rec_sessions_and_transformLs)

        #     offsetL.append(float(scaleD_cumulative[sig]['offset']))
        #     divisorL.append(float(scaleD_cumulative[sig]['divisor']) if scaleD_cumulative[sig]['divisor'] != 0 else 1.0)

        # self.offsetL  = offsetL
        # self.divisorL = divisorL

        return rec_sessions

    def _apply_scenario_selection(self,session):
        print('apply scenario selection! session',session.ctrlD['userId'])
        session.applyTransforms([['scenario_selection',{'scenarioL':['standing_on_face']}]])
        return session 

    def _generate_scenarios(self, sessions):
        def _generate_session_scenarios(session):
            sr = session.trace_srate
            # assert sr == 
            scenario_boundaryL = []
            scenario_start_event_index = 0

            for i in range(1,len(session.eventDf)):
                prev_event = session.eventDf.iloc[i-1]
                event = session.eventDf.iloc[i]

                if prev_event['index']+1 != event['index']:
                    scenario_end_event_index = i-1
                    # sanity check that it's more than 1 event long. 
                    # scenarios that are only 1 event long are usually "stop" events / events that we don't want to include
                    if scenario_start_event_index != i-1:
                        scenario_boundaryL.append(
                            {
                                'start':session.eventDf.iloc[scenario_start_event_index]['start'],
                                'end'  :session.eventDf.iloc[scenario_end_event_index]['end'],
                            }
                        )
                    scenario_start_event_index = i

            scenarios = []
            num_windows = 0
            for scenario_boundary in scenario_boundaryL:
                start_ms = scenario_boundary['start']
                end_ms   = scenario_boundary['end']
                start_sample = int(np.floor(start_ms / 1000 * sr))
                end_sample   = int(np.floor(end_ms   / 1000 * sr))

                scenarios.append(session.trace_yM[start_sample:end_sample,:])
                num_windows += (end_sample - start_sample - self.input_window_length_samples - self.output_window_length_samples)

            return scenarios, num_windows


        scenarios_and_window_counts = [ _generate_session_scenarios(session) for session in sessions ]
        num_windows = 0
        scenarios = []
        for scenario_and_window_count in scenarios_and_window_counts:
            num_windows += scenario_and_window_count[1] 
            scenarios.append(scenario_and_window_count[0])
        # scenarios, window_counts = [ _generate_session_scenarios(session) for session in sessions ]
        scenarios = [ scenario for dataset in scenarios for scenario in dataset ]  # unpack the list of lists into 1 list
        print('total nubmer of scenarios =',len(scenarios))
        print('total nubmer of windows =',  num_windows)
        return scenarios

    def _apply_train_test_val_split(self, scenarioL):
        trainL = []; num_train = 0
        testL  = []; num_test  = 0
        valL   = []; num_val   = 0

        # use random number generator to partition scenarios into train, test, and val groups
        for scenario in scenarioL:
            rand = random.random()
            if rand < self.percent_test:
                testL.append(scenario)
                num_test += 1
            elif rand < self.percent_test + self.percent_val:
                valL.append(scenario)
                num_val += 1
            else: # rand >= self.percent_test + self.percent_val
                trainL.append(scenario)
                num_train += 1

        self.num_val   = num_val
        self.num_test  = num_test
        self.num_train = num_train 

        return trainL, testL, valL

    def _make_dataset_from_windowL( self, scenarioL, batch_size, repeat ):
        # (signal, gesture_label, label_mid, label_return, row_start_offset, row_end_offset, mid_end_offset, return_end_offset)
        dataset = tf.data.Dataset.from_generator(lambda: scenarioL, 
                                                output_types=(tf.dtypes.as_dtype(np.float32)) #,
                                                #output_shapes=(tf.TensorShape( [None, self.num_signals] )) 
                                                ) 
        dataset = dataset.map(self._do_windowing)
        dataset = dataset.map(self._do_flatten)
        # dataset = dataset.map(self._do_windowing)
        # dataset = dataset.map(self._do_tiling)
        # dataset = dataset.map(self._labels_to_onehot)
        ## we unbatch because not all of the gestures have the same number of windows and we want to be able 
        # to split gesture over multiple batches
        # dataset = dataset.apply(tf.data.experimental.unbatch())
        dataset = dataset.shuffle(buffer_size=self.buffer_size)
        dataset = dataset.repeat(repeat)
        return dataset

    def _do_windowing(self,scenario):
        shape = tf.shape(scenario)[0]
        print(shape)
        input_sig  = tf.identity(scenario)
        output_sig = tf.identity(scenario)
        input_sig_windowed  = tf.contrib.signal.frame(input_sig[:shape-self.output_window_length_samples,:], self.input_window_length_samples,  1, axis=0, pad_end=False)
        output_sig_windowed = tf.contrib.signal.frame(output_sig[self.input_window_length_samples:      ,:], self.output_window_length_samples, 1, axis=0, pad_end=False)
        return (input_sig_windowed,output_sig_windowed)


    def _do_flatten(self,sig_in,sig_out):
        n_windows = tf.shape(sig_in)[0]
        return (tf.reshape(sig_in,[n_windows,-1]),tf.squeeze(sig_out))

def make_seq_2_seq_dataset(params):
    input_path = params.input_dir

    transformL = [
        ['signal_selection',    { 'sigL':['rot_x','rot_y','rot_z','rot_w'] } ] ,
        ['boseAR_scaling',      { } ],
        ['scenario_selection',{'scenarioL':['standing_on_face']}]
    ]
    if params.normalization == 'baseline':
        transformL.append(['zscore_normalize',    { 'sigL':['rot_x','rot_y','rot_z','rot_w'], 'relFl':True, 'medianFl':False, 'result_label':'rot_zscore_normalize' }])
    elif params.normalization == 'signal':
        transformL.append(['zscore_normalize',    { 'sigL':['rot_x','rot_y','rot_z','rot_w'], 'relFl':False, 'medianFl':False, 'result_label':'rot_zscore_normalize' }])

    window_hop_samples           = params.window_hop_samples
    input_window_length_samples  = params.input_window_length_samples
    output_window_length_samples = params.output_window_length_samples
    
    # buffer_size  = buffer_size
    batch_size   = 32
    percent_val  = 0.2
    percent_test = 0.2

    # read RecordingSessions
    from RecordingSession.RecordingSession import RecordingSession
    sessions = []

    filepath = input_path
    files = [f for f in os.listdir(filepath) if f.endswith('json')]

    train_scenarios = []
    train_windows_x = [] 
    train_windows_y = [] 
    test_scenarios  = []
    test_windows_x  = [] 
    test_windows_y  = [] 
    val_scenarios   = []
    val_windows_x   = [] 
    val_windows_y   = [] 

    scaleD = {}

    for file in files:
        session = RecordingSession(os.path.join(filepath,file))
        print('converting',session.ctrlD['userId'],'to RecordingSession')
        # sessions.append(rs)

        # transform data
        applyTransforms_return = session.applyTransforms(transformL)
        for key0 in applyTransforms_return.keys():
            if key0 == 'rot_zscore_normalize':
                normalize_return = applyTransforms_return['rot_zscore_normalize']
                for key1 in normalize_return.keys():
                    # import ipdb; ipdb.set_trace() 
                    if key1 not in scaleD.keys():
                        scaleD[key1] = {'offset':0,'divisor':0,'count':0}
                    scaleD[key1]['count']   += 1
                    scaleD[key1]['offset']  += float(normalize_return[key1]['offset'])
                    scaleD[key1]['divisor'] += float(normalize_return[key1]['divisor'])

        # import ipdb; ipdb.set_trace()
    
        # generate windows
        df = pd.DataFrame(columns=['file', 'scenario', 'scenario_start_index', 'scenario_stop_index', 'category'])
        # for session in sessions:
        print('generate windows! session',session.ctrlD['userId'])
        sr = session.trace_srate
        scenario_boundaryL = []
        scenario_start_event_index = 0

        for i in range(1,len(session.eventDf)):
            prev_event = session.eventDf.iloc[i-1]
            event = session.eventDf.iloc[i]

            if prev_event['index']+1 != event['index']:
                scenario_end_event_index = i-1
                # sanity check that it's more than 1 event long (aka, not just a stop event after a walking scenario)
                if scenario_start_event_index != i-1:
                    scenario_boundaryL.append(
                        {
                            'start':session.eventDf.iloc[scenario_start_event_index]['start'],
                            'end':  session.eventDf.iloc[scenario_end_event_index]['end']
                        }
                    )
                scenario_start_event_index = i

        # for scenario_boundary in scenario_boundaryL:
        for s in range(len(scenario_boundaryL)):
            scenario_boundary = scenario_boundaryL[s]
            rand = random.random()  # generate random number to partition into test, train, val

            start_ms = scenario_boundary['start']
            end_ms   = scenario_boundary['end']
            start_sample = int(np.floor(start_ms / 1000 * sr))
            end_sample   = int(np.floor(end_ms   / 1000 * sr))
            
            # write a csv with the data from this scenario
            if params.write_csv:
                print('writing csv!', session.ctrlD['userId'],'scenario number',s)
                # df.append({
                #     'file' : file, 
                #     'scenario' : 'standing_on_face', 
                #     'scenario_start_index' : start_sample, 
                #     'scenario_stop_index' : end_sample, 
                #     'category' : 'val' if rand < percent_val else 'test' if rand < percent_val + percent_test else 'train'
                # })
                filename = session.ctrlD['userId'] + '_' + str(s)
                filename += '_val' if rand < percent_val else '_test' if rand < percent_val + percent_test else '_train'
                np.save(file=os.path.join( os.path.join(params.output_dir,'data'), filename ),
                        arr=session.trace_yM[start_sample:end_sample,:])

            if rand < percent_val:
                val_scenarios.append(session.trace_yM[start_sample:end_sample,:])
                # val_windows_x.append(window)
                # val_windows_y.append(label)
            elif rand < percent_val + percent_test:
                test_scenarios.append(session.trace_yM[start_sample:end_sample,:])
                # test_windows_x.append(window)
                # test_windows_y.append(label)
            else:
                train_scenarios.append(session.trace_yM[start_sample:end_sample,:])
                # train_windows_x.append(window)
                # train_windows_y.append(label)


            """
            # i should hop entirely over the input window and the output window
            i = start_sample
            # hop = self.input_window_length_samples + self.output_window_length_samples
            while i < end_sample - (input_window_length_samples + output_window_length_samples):
                window = np.copy(session.trace_yM[i:i+input_window_length_samples,:])
                label  = np.copy(session.trace_yM[i+input_window_length_samples:i+input_window_length_samples+output_window_length_samples,:])
                window = np.squeeze(window.reshape((-1))) # feed a 1D vector
                # label  = np.squeeze(label); label = np.array(label[0])  
                # print('window.shape =',window.shape,'\tlabel.shape =',label.shape)
                if window.shape != (200,): # or label.shape != (4,):
                    print('window.shape =',window.shape,'\tlabel.shape =',label.shape)
                    import ipdb; ipdb.set_trace()
                # windows.append((window,label))
                if rand < percent_val:
                    val_windows_x.append(window)
                    val_windows_y.append(label)
                elif rand < percent_val + percent_test:
                    test_windows_x.append(window)
                    test_windows_y.append(label)
                else:
                    train_windows_x.append(window)
                    train_windows_y.append(label)
                # windows_x.append(window)
                # windows_y.append(label)
                i += 1 #input_window_length_samples + output_window_length_samples
            """
    
    def _window_scenarios(scenarios):
        windows_x = []
        windows_y = []
        for scenario in scenarios:
            i = 0
            while i < scenario.shape[0] - input_window_length_samples - output_window_length_samples:
                in_window  = scenario[i:i+input_window_length_samples,:]
                out_window = scenario[i+input_window_length_samples:i+input_window_length_samples+output_window_length_samples,:]
                if params.normalization == 'windows':
                    min_ = np.min([np.min(in_window),np.min(out_window)]) 
                    max_ = np.max([np.max(in_window),np.max(out_window)]) 
                    # min_max scale each signal
                    in_window  = (in_window  - min_) / (max_ - min_)
                    out_window = (out_window - min_) / (max_ - min_)
                in_window  = in_window.reshape((-1))
                windows_x.append(in_window)
                windows_y.append(out_window)
                i += window_hop_samples
        return (windows_x, windows_y)

    val_windows_x,   val_windows_y   = _window_scenarios(val_scenarios)
    test_windows_x,  test_windows_y  = _window_scenarios(test_scenarios)
    train_windows_x, train_windows_y = _window_scenarios(train_scenarios)

    print('num train =',len(train_windows_x),'\tnum test =',len(test_windows_x),'\tnum val =',len(val_windows_x))
    
    # import ipdb; ipdb.set_trace()

    returnD = {
        'train': {
            'x': train_windows_x,
            'y': train_windows_y,
            'scenarios':train_scenarios
        },
        'test': {
            'x': test_windows_x,
            'y': test_windows_y,
            'scenarios':test_scenarios
        },
        'val': {
            'x': val_windows_x,
            'y': val_windows_y,
            'scenarios':val_scenarios
        },
        'df': df,
        'scaleD': scaleD
    }

    return returnD

    # windows = self._generate_windows(rec_sessions)
    # train_windowL, test_windowL, val_windowL = self._apply_train_test_val_split(windows)
    # self.train_dataset = self._make_dataset_from_windowL(train_windowL, batch_size=self.batch_size, repeat=-1)
    # self.test_dataset  = self._make_dataset_from_windowL(test_windowL,  batch_size=1,               repeat=-1)
    # self.val_dataset   = self._make_dataset_from_windowL(val_windowL,   batch_size=1,               repeat=0)

