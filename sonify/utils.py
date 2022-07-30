from .plotter import _spectrogram
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from obspy import read

PAD = 60

class SeisPlotter(object):
    """
    Class that provides several solutions for plotting large and small waveform
    data sets.

    .. warning::

        This class should NOT be used directly, instead use the
        :meth:`~obspy.core.stream.Stream.plot` method of the
        ObsPy :class:`~obspy.core.stream.Stream` or
        :class:`~obspy.core.trace.Trace` objects.

    It uses matplotlib to plot the waveforms.
    """

    def __init__(self, client='', **kwargs):
        """
        Checks some variables and maps the kwargs to class variables.
        """
        self.kwargs = kwargs
        self.type = kwargs.get('type', None)
        self.file = kwargs.get('file', None)
        self.network = kwargs.get('network', '')
        self.station = kwargs.get('station', '')
        self.location = kwargs.get('location', '')
        self.channel = kwargs.get('channel', '')
        self.starttime = kwargs.get('starttime', '')
        self.endtime = kwargs.get('endtime', '')
        if self.type == 'fdsn':
         self.client = Client(client)
         self.st = self.client.get_waveforms(
           self.network,
           self.station,
           self.location,
           self.channel,
           self.starttime - PAD,
           self.endtime + PAD,
           attach_response=True,
        )
        elif self.type == 'file':
         self.st = read(self.file)
        self.st.merge(fill_value='interpolate')