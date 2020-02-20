"""
Functions to start and configure the Flask
application. This simply configures the server
and its routes.
"""
import os
import flask

from flask_caching import Cache
from flask_cors import CORS, cross_origin

from cryptonic import Model
#from cryptonic import CoinMarketCap
from cryptonic.api.routes import create_routes
import yfinance as yf

UI_DIST_DIRECTORY = os.getenv('UI_DIST_DIRECTORY', '../cryptonic-ui/dist/')


class Server:
    """
    Cryptonic server representation. This class
    contains logic for managing the configuration
    and deployment of Flask server.

    Parameters
    ----------
    debug: bool, default False
        If should start with a debugger.

    cors: bool, default True
        If the application should accept CORS
        requests.

    """
    def __init__(self, debug=False, cors=True):
        self.debug = debug
        self.cors = cors

        self.create_model()
        self.app = self.create()

    def create_model(self):
        """
        Creates a model either using a model provided
        by user or by creating a new model using
        previously researched parameters.

        Returns
        -------
        Trained Keras model. Ready to be used 
        via the model.predict() method.
        """
        ticker =  yf.Ticker("BTC-USD")
        historic_data = ticker.history(period='max')
        historic_data = historic_data.rename(columns={'Open':'open', 'High':'high', 'Low':'low', 'Close':'close', 'Volume':'volume'})
        historic_data.index.names = ['date']
        historic_data = historic_data[['open','high', 'low', 'close', 'volume']]
        historic_data = historic_data.reset_index()
        
        model_path = os.getenv('MODEL_NAME')

        #
        #  TODO: Figure out how large the data is for
        #  the old model and re-train. Maybe what I have
        #  to do here is to copy the weights of the
        #  model into a new model.
        #

        self.model = Model(data=historic_data,
                           path=model_path,
                           variable='close',
                           predicted_period_size=int(os.getenv('PERIOD_SIZE', 7)))

        if not model_path:
            self.model.build()
            self.model.train(epochs=int(os.getenv('EPOCHS', 50)), verbose=1)

        return self.model

    def create(self):
        """
        Method for creating a Flask server.

        Returns
        -------
        A Flask() application object.
        """
        app = flask.Flask(__name__, static_url_path='/', static_folder=UI_DIST_DIRECTORY)

        #
        #  Application configuration. Here we
        #  configure the application to accept
        #  CORS requests, its routes, and
        #  its debug flag.
        #
        if self.cors:
            CORS(app)

        cache_configuration = {
            'CACHE_TYPE': 'redis',
            'CACHE_REDIS_URL': os.getenv('REDIS_URL',
                                         "redis://localhost:6379/2")
        }

        self.cache = Cache(app, config=cache_configuration)

        app.config['DEBUG'] = self.debug
        create_routes(app, self.cache, self.model)

        return app

    def run(self, *args, **kwargs):
        """
        Method for running Flask server.
        Parameters
        ----------
        *args, **kwargs: parameters passed to the Flask application. 
        """
        self.app.run(*args, **kwargs)