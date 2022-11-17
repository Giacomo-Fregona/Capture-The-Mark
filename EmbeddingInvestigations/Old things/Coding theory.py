import numpy as np
import unireedsolomon as rs


character_encoding = 'utf-8'
bytestring= bytes([10])
print(bytestring)
unicode_text = str(bytestring, character_encoding)

coder = rs.RSCoder(255,128)
c= coder.encode(unicode_text)
print("repr c: " + repr(c))
decoded = coder.decode(c)[0]
print("decoded: "+ decoded )


byte = bytes(decoded, 'utf-8')

print(byte)










