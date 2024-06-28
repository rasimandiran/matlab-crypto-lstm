function E = xentropy(d, y)

E = sum(-d.*log(y)-(1-d).*log(1-y));

end