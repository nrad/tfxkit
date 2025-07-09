from icecube import ml_suite, icetray
from collections import deque
import timeit
import numpy as np


class MyModelWrapper(ml_suite.TFModelWrapper):

    def __init__(self, context):
        super().__init__(context)
        self.AddParameter("n_inputs", "Number of input features")
        self.AddParameter(
            "event_stream",
            "Name of the event stream to process",
            icetray.I3Frame.Physics,
        )
        self.AddParameter(
            "preprocessor",
            "Function to preprocess the input batch before prediction",
        )
        # icetray.I3ConditionalModule.__init__(self, context)
        # self.AddParameter("cfg_file", "", None)
        # self.AddParameter("output_key", "", "ml_suite_features")
        # self.AddParameter("plot_results", "Plot results if true.", False)

    def Configure(self):
        """Configure module."""
        nn_model = self.GetParameter("nn_model")
        self._batch_size = self.GetParameter("batch_size")
        self._output_key = self.GetParameter("output_key")
        self._write_runtime_info = self.GetParameter("write_runtime_info")
        self._output_names = self.GetParameter("output_names")
        self._data_transformer = self.GetParameter("data_transformer")
        self._event_feature_extractor = self.GetParameter("event_feature_extractor")
        self._sub_event_stream = self.GetParameter("sub_event_stream")
        self._event_stream = self.GetParameter("event_stream")
        self._n_inputs = self.GetParameter("n_inputs")
        self._preprocessor = self.GetParameter("preprocessor")

        n_inputs = self._n_inputs
        # self._event_input_shapes = [[n_inputs]]
        self._nested_input_tensors = False

        if self._data_transformer is not None:
            raise NotImplementedError("data_transformer not implemented")

        print(f"------------------\n{dir(self) = }\n{self.__dict__ = }")
        print(
            f"""
        {self._batch_size = }
        {self._output_key = }
        {self._write_runtime_info = }
        {self._output_names = }
              
              """
        )

        # configure the model
        self._configure_model()

        # create variables and frame buffer for batching
        self._frame_buffer: "deque[icetray.I3Frame]" = deque()
        self._pframe_counter = 0
        self._batch_event_index = 0
        self._input_batch_raw = np.zeros([self._batch_size], dtype="object")
        self._runtime_batch = np.empty([self._batch_size])
        self._y_pred_batch = None
        self._runtime_prediction = None

    def _process_frame_buffer(self):
        """
        Performs prediction for accumulated batch.
        Then writes results to physics frames in frame buffer and eventually
        pushes all of the frames in the order they came in.
        """
        import utils

        if self._preprocessor is not None:
            preprocessing_func = self._preprocessor
            self._input_batch = preprocessing_func(self._input_batch_raw)
        else:
            self._input_batch = self._input_batch_raw

        # self._input_batch = preprocessing_func(self._input_batch_raw)

        self._perform_prediction(size=self._pframe_counter)

        # reset counters and indices
        self._batch_event_index = 0
        self._pframe_counter = 0

        # push frames
        while self._frame_buffer:
            fr = self._frame_buffer.popleft()

            if (fr.Stop == self._event_stream) and (
                (self._sub_event_stream is None)
                or (fr["I3EventHeader"].sub_event_stream == self._sub_event_stream)
            ):

                # write results at current batch index to frame
                self._write_to_frame(fr, self._batch_event_index)

                # increase the batch event index
                self._batch_event_index += 1

            self.PushFrame(fr)

    def Process(self):
        """Process incoming frames.

        Pop frames and put them in the frame buffer.
        When a physics frame is popped, accumulate the input data to form
        a batch of events. Once a full batch of physics events is accumulated,
        perform the prediction and push the buffered frames.
        The Physics method can then write the results to the physics frame
        by using the results: ::

            self._y_pred_batch
            self._runtime_prediction, self._runtime_preprocess_batch

        and the current event index self._batch_event_index
        """
        frame = self.PopFrame()

        # put frame on buffer
        self._frame_buffer.append(frame)

        # check if the current frame is a physics frame
        if (frame.Stop == self._event_stream) and (
            (self._sub_event_stream is None)
            or (frame["I3EventHeader"].sub_event_stream == self._sub_event_stream)
        ):

            # add input data for this event
            start_time = timeit.default_timer()

            # get (possibly transformed) data
            data = self._get_data(frame)
            self._input_batch_raw[self._pframe_counter] = data
            self._runtime_batch[self._pframe_counter] = (
                timeit.default_timer() - start_time
            )

            self._pframe_counter += 1

            # check if we have a full batch of events
            if self._pframe_counter == self._batch_size:

                # we have now accumulated a full batch of events so
                # that we can perform the prediction
                self._process_frame_buffer()
