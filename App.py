import tornado.ioloop
import tornado.web
from tornado.escape import json_decode
from tornado import gen
from scipy.optimize import minimize
import numpy as np
import time
import plotly.graph_objs as go
import base64

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")

class CalculateHandler(tornado.web.RequestHandler):
    @gen.coroutine
    def post(self):
        data = json_decode(self.request.body)
        Xa = float(data['Xa'])
        T = float(data['T'])
        D0_ab = float(data['D0_ab'])
        D0_ba = float(data['D0_ba'])
        ra = float(data['ra'])
        rb = float(data['rb'])
        qa = float(data['qa'])
        qb = float(data['qb'])
        D_exp = float(data['D_exp'])

        # Définir Xb en fonction de Xa
        Xb = 1 - Xa
        # Définir une fonction pour calculer Dab en fonction de Xa, Aab et Aba
        def Dab_calc(Xa, Aab, Aba):
            term1 = Xb*np.log(D0_ab) + Xa*D0_ba + 2*(Xa*np.log((Xa*ra+Xb*rb)/ra)+Xb*np.log((Xb*rb+Xa*ra)/rb))
            term2 = 2*Xa*Xb*((ra/(Xa*ra+Xb*rb))*(1-(ra/rb)) + (rb/(Xa*ra+Xb*rb))*(1-(rb/ra)))
            term3 = Xb*qa*((1-((Xa*qa*np.exp(-Aba/T))/(Xa*qa+Xb*qb*np.exp(-Aba/T)))**2)*(-Aba/T)+(1-((Xb*qb)/(Xb*qb+Xa*qa*np.exp(-Aab/T))))**2*np.exp(-Aab/T)*(-Aba/T))
            term4 = Xa*qb*((1-((Xb*qb*np.exp(-Aab/T))/(Xa*qa*np.exp(-Aab/T)+Xb*qb))**2)*(-Aab/T)+(1-((Xa*qa)/(Xa*qa+Xb*qb*np.exp(-Aba/T))))**2*np.exp(-Aba/T)*(-Aab/T))
    # Calcul de D_AB
            D_AB = np.exp(term1 + term2 + term3 + term4)
            return D_AB
     
        # Définir une fonction pour calculer l'erreur à minimiser
        def error_func(A):
            return abs(Dab_calc(Xa, A[0], A[1]) - D_exp)

        # Initialiser Aab et Aba
        A = np.array([0, 400])
        tol = 1e-14  # Tolérance pour la différence entre Dab(th) et Dab(exp)
        num_iterations = 0
        start_time = time.time()
        Dab_th = Dab_calc(Xa, A[0], A[1])  # Calcul initial de Dab_th

        while True:
            num_iterations += 1
            if abs(Dab_th - D_exp) < tol:
                break
            result = minimize(error_func, A, method='Nelder-Mead')  # Minimisation de la fonction d'erreur
            A = result.x
            Dab_th = Dab_calc(Xa, A[0], A[1])

        end_time = time.time()
        execution_time = end_time - start_time

        # Tracer le graphe
        Xa_values = np.linspace(0, 0.7, 100)  # Fraction molaire de A
        D_AB_values = [Dab_calc(Xa, A[0], A[1]) for Xa in Xa_values]

        fig = go.Figure(data=go.Scatter(x=Xa_values, y=D_AB_values))
        fig.update_layout(
            title='Variation du coefficient de diffusion en fonction de la fraction molaire',
            xaxis=dict(title='Fraction molaire de A'),
            yaxis=dict(title='Coefficient de diffusion (cm^2/s)'),
            hovermode='closest'
        )

        # Convertir le graphe en image
        img_bytes = fig.to_image(format="png")
        img_data = base64.b64encode(img_bytes).decode("utf-8")

        # Retourner les résultats au format JSON
        response_data = {
            "nombre_iterations": num_iterations,
            "temps_execution": execution_time,
            "Aab_optimal": A[0],
            "Aba_optimal": A[1],
            "Dab_th": Dab_th,
            "ecart_minimum": abs(Dab_th - D_exp),
            "graph_image": img_data  # Envoyer l'image encodée en base64
        }

        self.write(response_data)

def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/calculate", CalculateHandler),
    ], template_path="templates")

if __name__ == "__main__":
    app = make_app()
    app.listen(8888)
    tornado.ioloop.IOLoop.current().start()
