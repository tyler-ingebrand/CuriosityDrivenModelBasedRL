abstract type Agent

end


(::Agent)(state) = Error("Unimplemented")
learn!(::Agent, experience) = pass