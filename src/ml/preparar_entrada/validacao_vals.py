import ast

class Validacao:

    @staticmethod
    def valor_valido(valor):
        # print(valor)
        if valor is None or str(valor).strip() == '':
            return False
        if isinstance(valor, str):
            valor = str(valor).replace(',', '.')
        try:
            float(valor)
            return True
        except ValueError:
            return False
        
    @staticmethod
    def eval_valido(valor):
        try:
            return ast.literal_eval(valor)
        except (ValueError, SyntaxError) as e:
            print(f"[ERRO: {e}]")
            return (0.0, 0.0)