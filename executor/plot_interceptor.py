import io
import base64
import sys


class PlotInterceptor:
    def __init__(self):
        self.plots = []
        self._original_show = None

    def install(self):
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt

            self._original_show = plt.show

            def captured_show(*args, **kwargs):
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                buf.seek(0)
                img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                self.plots.append(img_base64)
                buf.close()
                plt.close('all')

            plt.show = captured_show
        except ImportError:
            pass

    def get_plots(self):
        return self.plots


# Global interceptor instance
interceptor = PlotInterceptor()
