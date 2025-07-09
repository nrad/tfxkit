from icecube import icetray, dataclasses, dataio
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# class PrimaryParticleCheckerModule(icetray.I3Module):
#     """ 
#     This module checks if the primary particle in the input file is present in the target file.
#     If the primary particle is found, the event is accepted with a probability of `AcceptanceProbability`.
#     If the primary particle is not found, the event is rejected.

#     Here the minorID and majorID are actually ignored, and the match is done based on the energy, direction, position and type of the particle.
    
#     """

#     def __init__(self, context):
#         super().__init__(context)
#         self.AddParameter("InputFile", "Path to the input file", "")
#         self.AddParameter("TargetFile", "Path to the target file", "")
#         self.AddParameter("AcceptanceProbability", "Probability to accept the event if primary particle is found", 0.5)
#         self.AddParameter("InputKey", "Input key for the primary particle", "I3MCTree")
#         self.AddParameter("TargetKey", "Target key for the primary particle", "I3MCTree_preMuonProp")
#         self.AddParameter("ProcessingWeightKey", "Key for the processing weight", "ProcessingWeight")
#         self.AddOutBox("OutBox")

#     def Configure(self):
#         self.input_file = self.GetParameter("InputFile")
#         self.target_file = self.GetParameter("TargetFile")
#         self.acceptance_probability = self.GetParameter("AcceptanceProbability")
#         self.input_key = self.GetParameter("InputKey")
#         self.target_key = self.GetParameter("TargetKey")
#         self.processing_weight_key = self.GetParameter("ProcessingWeightKey")
#         self.target_data = dataio.I3File(self.target_file)
#         self.target_primary_particles = self._get_primary_particles(self.target_data, key=self.target_key)

#         print(f"{len(self.target_primary_particles) = }")

#     def DAQ(self, frame):
#         if self.input_key not in frame:
#             self.PushFrame(frame)
#             return

#         primary_particle = frame[self.input_key].primaries[0]
#         primary_info = extract_particle_info(primary_particle)
        
#         if primary_info in self.target_primary_particles:
#             self.target_primary_particles.pop(self.target_primary_particles.index(primary_info))            
#             ProcessingWeight = 1.0
#         else:
#             ProcessingWeight = 1./self.acceptance_probability if self.acceptance_probability else 0
#             if np.random.random() > self.acceptance_probability:
#                 return # Reject the Event

#         frame[self.processing_weight_key] = dataclasses.I3Double(ProcessingWeight)
#         self.PushFrame(frame)


#     def Finish(self):
#         if self.target_primary_particles:
#             raise RuntimeError(f"Primary particles not found in the target file: {self.target_primary_particles = }")
#         else:
#             print("All primary particles found in the target file!")
#         self.target_data.close()

#     def _get_primary_particles(self, data_file, key="I3MCTree_preMuonProp", event_stream=icetray.I3Frame.DAQ):
#         primary_particles = list()
#         while data_file.more():
#             frame = data_file.pop_frame()
#             if not frame.Stop == event_stream:
#                 continue
#             primary_particle = frame[key].primaries[0]
#             primary_particles.append(extract_particle_info(primary_particle))
#         return primary_particles


class PrimaryParticleMatcherModule(icetray.I3Module):
    """ 
    This module checks if the primary particle in the input file is present in the target file.
    If the primary particle is found, the event is accepted with a probability of `AcceptanceProbability`.
    If the primary particle is not found, the event is rejected.

    Here the minorID and majorID are actually ignored, and the match is done based on the energy, direction, position and type of the particle.

    IMPORTANT: The primaries are assumed to be in the same order in both the input and target files.
    
    """

    def __init__(self, context):
        super().__init__(context)
        self.AddParameter("TargetFile", "Path to the target file", "")
        self.AddParameter("InputKey", "Input key for the primary particle", "I3MCTree")
        self.AddParameter("TargetKey", "Target key for the primary particle", "I3MCTree_preMuonProp")
        self.AddParameter("AcceptanceProbability", "Probability to accept the event if primary particle is found", 0.5)
        self.AddParameter("ProcessingWeightKey", "Key for the processing weight", "ProcessingWeight")
        self.AddParameter("Strict", "Raise error if not all primary particles in target file having matching ones in the input",False)
        self.AddOutBox("OutBox")

    def Configure(self):
        self.input_key = self.GetParameter("InputKey")
        self.target_file = self.GetParameter("TargetFile")
        self.target_key = self.GetParameter("TargetKey")
        self.acceptance_probability = self.GetParameter("AcceptanceProbability")
        self.processing_weight_key = self.GetParameter("ProcessingWeightKey")
        self.strict = self.GetParameter("Strict")
        #self.target_data = dataio.I3File(self.target_file)
        # self.target_primary_particles = self._get_primary_particles(self.target_data, key=self.target_key)
        self.primary_yielder = yield_primary_particles(self.target_file, key=self.target_key)


        # self.target_primary = next(self.primary_yielder)
        try:
            self.target_primary = next(self.primary_yielder)
            #print("--"*8)
        except StopIteration:
            self.target_primary = None
        # print(f"{len(self.target_primary_particles) = }")

    def DAQ(self, frame):
        if self.input_key not in frame:
            return
        if not self.target_primary:
            #print("No more primary particles to match!")
            return 

        primary_particle = frame[self.input_key].primaries[0]
        primary_info = extract_particle_info(primary_particle)

        # target_primary = next(self.primary_yielder)
        # target_primary = self.target_primary

        if primary_info == self.target_primary:                
            ProcessingWeight = 1.0
            try:
                self.target_primary = next(self.primary_yielder)
                #print("--"*8)
            except StopIteration:
                self.target_primary = None
                
        else:
            ProcessingWeight = 1./self.acceptance_probability if self.acceptance_probability else 0
            if np.random.random() > self.acceptance_probability:
                return

        frame[self.processing_weight_key] = dataclasses.I3Double(ProcessingWeight)
        #print(f"{primary_info}")
        self.PushFrame(frame)
        return

    def _DAQ(self, frame):
        if self.input_key not in frame:
            return

        primary_particle = frame[self.input_key].primaries[0]
        primary_info = extract_particle_info(primary_particle)
        
        if primary_info in self.target_primary_particles:
            primary_index = self.target_primary_particles.index(primary_info)
            # print(primary_index)
            assert primary_index == 0, primary_index
            self.target_primary_particles.pop(primary_index)            
            ProcessingWeight = 1.0
        else:
            ProcessingWeight = 1./self.acceptance_probability if self.acceptance_probability else 0
            if np.random.random() > self.acceptance_probability:
                return

        frame[self.processing_weight_key] = dataclasses.I3Double(ProcessingWeight)
        self.PushFrame(frame)
        return


    def Finish(self):
        remaining_primaries = list(self.primary_yielder)
        if remaining_primaries:
            comment = f"Primary particles not found in the target file: {len(remaining_primaries) = }"
            if self.strict:
                raise RuntimeError(comment)
            else:
                print("\n!!! WARNING !!!", comment, "\n")
        else:
            print("All primary particles found in the target file!")
        # if self.target_primary_particles:
        #     raise RuntimeError(f"Primary particles not found in the target file: {self.target_primary_particles = }")
        # else:
        #     print("All primary particles found in the target file!")
        #self.target_data.close()

    # def _get_primary_particles(self, data_file, key="I3MCTree_preMuonProp", event_stream=icetray.I3Frame.DAQ):
    #     primary_particles = list()
    #     while data_file.more():
    #         frame = data_file.pop_frame()
    #         if not frame.Stop == event_stream:
    #             continue
    #         primary_particle = frame[key].primaries[0]
    #         primary_particles.append(extract_particle_info(primary_particle))
    #     return primary_particles

def yield_primary_particles(target_files, key="I3MCTree_preMuonProp", event_stream=icetray.I3Frame.DAQ):
    if not isinstance(target_files, list):
        target_files = [target_files]

    i = 0
    for current_file in target_files:
        data_file = dataio.I3File(current_file)
        while data_file.more():
            frame = data_file.pop_frame()
            if not frame.Stop == event_stream:
                continue
            primary_particle = frame[key].primaries[0]
            primary_info = extract_particle_info(primary_particle)
            # print(f"----" * 8)
            # print(f"Yielding: {i} \n{primary_info}")
            i+=1
            yield primary_info
        data_file.close()


def match_particles(particle1, particle2, rtol=1e-5):
    energy_match = np.isclose(particle1.energy, particle2.energy, rtol=rtol)
    dir_x_match = np.isclose(particle1.dir.x, particle2.dir.x, rtol=rtol)
    dir_y_match = np.isclose(particle1.dir.y, particle2.dir.y, rtol=rtol)
    dir_z_match = np.isclose(particle1.dir.z, particle2.dir.z, rtol=rtol)
    pos_x_match = np.isclose(particle1.pos.x, particle2.pos.x, rtol=rtol)
    pos_y_match = np.isclose(particle1.pos.y, particle2.pos.y, rtol=rtol)
    pos_z_match = np.isclose(particle1.pos.z, particle2.pos.z, rtol=rtol)
    type_match = particle1.type == particle2.type

    return (energy_match and dir_x_match and dir_y_match and dir_z_match and
            pos_x_match and pos_y_match and pos_z_match and type_match)


def extract_particle_info(particle):
    return {
        "energy": particle.energy,
        "dir_x": particle.dir.x,
        "dir_y": particle.dir.y,
        "dir_z": particle.dir.z,
        "pos_x": particle.pos.x,
        "pos_y": particle.pos.y,
        "pos_z": particle.pos.z,
        "type": particle.type
    }

if __name__ == "__main__":

    parser = ArgumentParser(description="Primary Particle Checker", formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_file", required=True, help="Path to the input file")
    parser.add_argument("--target_file", required=True, help="Path to the target file", nargs="+")
    parser.add_argument("--output_file", required=True, help="Path to the output file")
    parser.add_argument("--acceptance_probability", type=float, default=0.5, help="Probability to accept the event if primary particle is found")

    args = parser.parse_args()

    tray = icetray.I3Tray()
    tray.AddModule("I3Reader", "reader", Filename=args.input_file)
    tray.AddModule(PrimaryParticleMatcherModule, "primary_checker",
                   #InputFile=args.input_file,
                   TargetFile=args.target_file,
                   AcceptanceProbability=args.acceptance_probability)
    tray.AddModule("I3Writer", "writer", Filename=args.output_file)
    tray.Execute()
    tray.Finish()


