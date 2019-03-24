import os
import tempfile
import datetime
import pydicom
from numpy import ComplexWarning
from PIL import Image
from pydicom.dataset import Dataset, FileDataset
from pydicom.filewriter import correct_ambiguous_vr
from pydicom.uid import ExplicitVRLittleEndian


def create_dcm_file(image, name="", study_date="", study_desc=""):
    filename_little_endian = tempfile.NamedTemporaryFile(suffix='.dcm').name

    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
    file_meta.MediaStorageSOPInstanceUID = "1.2.3"
    file_meta.ImplementationClassUID = "1.2.3.4"

    ds = FileDataset(filename_little_endian, {},
                     file_meta=file_meta, preamble=b"\0" * 128)

    ds.PatientName = name
    ds.StudyDate = study_date
    ds.StudyDescription = study_desc
    ds.PixelRepresentation = 1
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 1
    ds.HighBit = 15
    ds.BitsStored = 16
    ds.BitsAllocated = 16
    ds.SmallestImagePixelValue = str.encode('\x00\x00')
    ds.LargestImagePixelValue = str.encode('\xff\xff')
    ds.Columns = image.shape[0]
    ds.Rows = image.shape[1]

    dt = datetime.datetime.now()
    ds.ContentDate = dt.strftime('%Y%m%d')
    time_str = dt.strftime('%H%M%S.%f')
    ds.ContentTime = time_str

    if image.max() <= 1:
        image *= 255
        try:
            image = image.astype("uint16")
        except ComplexWarning:
            return None
    ds.PixelData = Image.fromarray(image).tobytes()

    try:
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = correct_ambiguous_vr(ds, True)
        ds.save_as(filename_little_endian)
    except AttributeError:
        return None

    try:
        ds = pydicom.dcmread(filename_little_endian)
        os.remove(filename_little_endian)
        return ds
    except OSError:
        return None
