# =====================================================================
# SIMULADOR DEL TALLER DE LEGOS: VERSIÓN CORREGIDA Y LIMPIA PARA CLASE
# =====================================================================

print("=========================================================")
print(" 🏎️  ¡BIENVENIDOS AL TALLER DE LEGOS DE INTELIGENCIA ARTIFICIAL!")
print("=========================================================\n")

# 📥 ENTRADA (Datos Crudos)
x = 5.0
print("📥 [ENTRADA] El cliente nos entrega:", x, "bloques de chasis básico.\n")

print("---------------------------------------------------------")
print("🛠️  PARTE 1: PROPAGACIÓN HACIA ADELANTE (Construcción del Juguete)")
print("---------------------------------------------------------")

# CONSTRUCTOR 1: Ejes y llantas
w1 = 3.0
b1 = 2.0
z1 = (x * w1) + b1
print("🧱 [Constructor 1]: Triplica piezas (*3) y suma rines (+2).")
print("    -> Cálculo: (5.0 * 3.0) + 2.0 =", z1)
print("    -> Entrega al compañero un chasis con llantas de", z1, "piezas.\n")

# CONSTRUCTOR 2: Motor y cofre
w2 = 2.0
b2 = 1.0
z2 = (z1 * w2) + b2
print("⚙️ [Constructor 2]: Duplica tamaño (*2) y suma un alerón (+1).")
print("    -> Cálculo: (17.0 * 2.0) + 1.0 =", z2)
print("    -> Entrega al compañero un carro casi listo de", z2, "piezas.\n")

# CONSTRUCTOR 3: Stickers y piloto
w3 = 2.0
b3 = 4.0
z3 = (z2 * w3) + b3
print("🏁 [Constructor 3]: Duplica tamaño (*2) y suma luces brillantes (+4).")
print("    -> Cálculo: (35.0 * 2.0) + 4.0 =", z3)
print("    -> OUT: ¡El carro sale terminado con", z3, "piezas!\n")

# 📤 EVALUACIÓN DEL CLIENTE
valor_manual = 79.0
error_final = valor_manual - z3
print("📤 [SALIDA DE LA RED] Carro terminado de", z3, "piezas.")
print("⚠️  [MANUAL OFICIAL] Exige exactamente:", valor_manual, "piezas.")
print("❌ [ERROR DETECTADO] Al taller le faltaron:", error_final, "bloques.\n")

print("---------------------------------------------------------")
print("📞 PARTE 2: RETROPROPAGACIÓN (El Teléfono Descompuesto)")
print("---------------------------------------------------------")
print("El jefe viaja hacia atrás reportando el error en la fila.")
print("Por la distancia entre mesas, el mensaje se reduce a la quinta parte.\n")

# El factor de pérdida del mensaje por el "pasillo"
factor_descomposicion = 0.2

# JEFE -> CONSTRUCTOR 3
mensaje_c3 = error_final
print("🗣️  [Jefe habla con el Constructor 3 - Última Mesa]:")
print("    -> Mensaje recibido:", mensaje_c3, "(Escucha el error entero).")
print("    -> REACCIÓN: '¡Entendido! Mañana pondré más luces y stickers'.\n")

# CONSTRUCTOR 3 -> CONSTRUCTOR 2
mensaje_c2 = mensaje_c3 * factor_descomposicion
print("📨 [Constructor 3 le avisa al Constructor 2 - Mesa Intermedia]:")
print("    -> Mensaje viaja y se encoge:", mensaje_c3, "*", factor_descomposicion, "=", mensaje_c2)
print("    -> Mensaje recibido:", mensaje_c2, "(Solo le llega un reclamo de 1 bloque).")
print("    -> REACCIÓN: 'Ok, creo que escuché que faltó algo en el motor...'.\n")

# CONSTRUCTOR 2 -> CONSTRUCTOR 1
mensaje_c1 = mensaje_c2 * factor_descomposicion
print("🤫 [Constructor 2 le susurra al Constructor 1 - Primera Mesa]:")
print("    -> Mensaje viaja y se encoge:", mensaje_c2, "*", factor_descomposicion, "=", mensaje_c1)
print("    -> Mensaje recibido:", mensaje_c1)
print("    -> REACCIÓN: '¿0.2 bloques? Bah, eso es casi nada.")
print("                 ¡Mi trabajo con las llantas quedó perfecto!')\n")

print("---------------------------------------------------------")
print("🎯 CONCLUSIÓN PEDAGÓGICA PARA LA CLASE")
print("---------------------------------------------------------")
print("Como el mensaje final que llegó al Constructor 1 fue de solo", mensaje_c1, ",")
print("él NO ajustará sus parámetros para el siguiente intento.")
print("¡El error se originó desde la base, pero el algoritmo nunca pudo corregirlo!")
print("A esto los científicos le llaman DESVANECIMIENTO DEL GRADIENTE.")
print("=========================================================")
