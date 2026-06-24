"""High-level face-recognition pipeline."""

from __future__ import annotations

import importlib.util
import os
import warnings
from collections.abc import Sequence
from typing import Any, Literal, cast, overload

import numpy as np
import numpy.typing as npt

from lvface.detect import FaceDetector, InsightFaceDetector
from lvface.embed import FaceEmbedder, LVFaceOnnxEmbedder
from lvface.errors import NoFaceError
from lvface.hub import resolve_weights
from lvface.io import ImageInput, load_image
from lvface.metrics import DEFAULT_THRESHOLDS, dot, euclidean, pairwise_cosine, to_percentage
from lvface.registry import DEFAULT_MODEL
from lvface.runtime import Device
from lvface.types import BBox, ComparisonResult, Embedding, Face, Match, MatchResult

Selector = Literal["largest", "highest_score", "error"]
Assignment = Literal["greedy", "hungarian", "all"]
ImageCollection = ImageInput | Sequence[ImageInput]


def _device(device: str) -> Device:
    """Validate a requested inference device.

    Args:
        device: Device name supplied by the caller.

    Returns:
        The validated device literal.
    """
    if device not in {"auto", "cpu", "cuda", "directml"}:
        raise ValueError("device must be 'auto', 'cpu', 'cuda', or 'directml'")

    return cast(Device, device)


def _make_embedder(
    embedder: str | os.PathLike[str] | FaceEmbedder,
    device: Device,
) -> FaceEmbedder:
    """Resolve an embedder instance or model reference.

    Args:
        embedder: Embedder instance, registered model name, or model path.
        device: Device used for a newly created embedder.

    Returns:
        A configured face embedder.
    """
    if isinstance(embedder, FaceEmbedder):
        return embedder

    if not isinstance(embedder, (str, os.PathLike)):
        raise TypeError("embedder must be a model name, path, or FaceEmbedder")

    return LVFaceOnnxEmbedder(resolve_weights(embedder), device=device)


def _make_detector(detector: str | FaceDetector | None, device: Device) -> FaceDetector | None:
    """Resolve an optional detector configuration.

    Args:
        detector: Detector instance, ``"insightface"``, or ``None``.
        device: Device used for a newly created detector.

    Returns:
        A configured detector or ``None``.
    """
    if detector is None or isinstance(detector, FaceDetector):
        return detector

    if detector != "insightface":
        raise ValueError(f"unknown face detector: {detector!r}")

    if importlib.util.find_spec("insightface") is None:
        raise ImportError('InsightFace detection requires `pip install "lvface[detect]"`')

    return InsightFaceDetector(device=device)


def _cosine_metric(metric: str) -> str:
    """Validate that cosine is used as the decision metric.

    Args:
        metric: Requested decision metric.

    Returns:
        The validated ``"cosine"`` value.
    """
    if metric != "cosine":
        raise ValueError("metric must be 'cosine'; distance metrics are diagnostic only")

    return metric


def _threshold(value: float | None) -> float:
    """Validate or supply a cosine decision threshold.

    Args:
        value: Explicit threshold or ``None`` for the default.

    Returns:
        A finite threshold in the range [-1, 1].
    """
    result = DEFAULT_THRESHOLDS["cosine"] if value is None else float(value)
    if not np.isfinite(result) or not -1.0 <= result <= 1.0:
        raise ValueError("threshold must be finite and between -1 and 1")

    return float(result)


def _sources(value: ImageCollection) -> list[ImageInput]:
    """Normalize one image source or a sequence into a list.

    Args:
        value: Single image source or sequence of sources.

    Returns:
        Image sources in input order.
    """
    if isinstance(value, bytes):
        return [value]

    if isinstance(value, (str, os.PathLike, np.ndarray)):
        return [value]

    if not isinstance(value, Sequence):
        raise TypeError("image input must be an image source or a sequence of image sources")

    return list(cast(Sequence[ImageInput], value))


def _face_key(face: Face) -> tuple[int, int]:
    """Return a stable ordering key for a face.

    Args:
        face: Face to identify.

    Returns:
        Its image and face indices.
    """
    return face.image_index, face.face_index


def _descending_score(score: float) -> float:
    """Negate a score for ascending sorts.

    Args:
        score: Similarity score.

    Returns:
        The negated score.
    """
    return -score


class FaceRecognizer:
    """Detect, align, embed, and compare faces."""

    def __init__(
        self,
        embedder: str | os.PathLike[str] | FaceEmbedder = DEFAULT_MODEL,
        detector: str | FaceDetector | None = "insightface",
        *,
        device: str = "auto",
        threshold: float | None = None,
        metric: str = "cosine",
    ) -> None:
        """Configure the face-recognition pipeline.

        Args:
            embedder: Embedder instance, model name, or ONNX path.
            detector: Detector instance, ``"insightface"``, or ``None`` for aligned inputs.
            device: Preferred inference device.
            threshold: Default cosine match threshold.
            metric: Decision metric; currently only ``"cosine"`` is supported.
        """
        resolved_device = _device(device)
        self.embedder = _make_embedder(embedder, resolved_device)
        self.detector = _make_detector(detector, resolved_device)
        self.metric = _cosine_metric(metric)
        self.threshold = _threshold(threshold)

    def analyze(self, src: ImageInput) -> list[Face]:
        """Detect, align, and embed every usable face in an image.

        Args:
            src: Image path, URL, bytes, or array.

        Returns:
            Analyzed faces with normalized embeddings.
        """
        image = load_image(src)
        if self.detector is None:
            embedding = self.embedder.embed(image)
            return [Face(None, None, image, embedding)]

        self.detector.load()
        detected_faces = self.detector.detect(image)
        aligned_faces: list[Face] = []
        crops: list[npt.NDArray[np.uint8]] = []

        for face_index, detected_face in enumerate(detected_faces):
            face = Face(
                bbox=detected_face.bbox,
                kps=detected_face.kps,
                aligned=None,
                embedding=None,
                face_index=face_index,
            )
            if face.kps is None:
                warnings.warn(
                    f"skipping face {face.face_index}: five-point landmarks are missing",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue

            face.aligned = self.detector.align(image, face.kps)
            aligned_faces.append(face)
            crops.append(face.aligned)

        embeddings = self.embedder.embed_batch(crops)
        for face, embedding in zip(aligned_faces, embeddings, strict=True):
            face.embedding = embedding

        return aligned_faces

    @overload
    def embed(self, src: ImageInput, *, select: None = None) -> list[Embedding]:
        """Embed every face when no selector is provided.

        Args:
            src: Image path, URL, bytes, or array.
            select: Must be ``None`` to return every embedding.
        """
        ...

    @overload
    def embed(self, src: ImageInput, *, select: Selector) -> Embedding:
        """Embed one face selected by policy.

        Args:
            src: Image path, URL, bytes, or array.
            select: Policy for choosing one detected face.
        """
        ...

    def embed(
        self,
        src: ImageInput,
        *,
        select: Selector | None = None,
    ) -> Embedding | list[Embedding]:
        """Embed every face or select one by policy.

        Args:
            src: Image path, URL, bytes, or array.
            select: Optional policy for selecting one detected face.

        Returns:
            All embeddings, or one selected embedding.
        """
        faces = self.analyze(src)
        if select is None:
            return [face.embedding for face in faces if face.embedding is not None]

        embedding = self._select(faces, select).embedding
        if embedding is None:
            raise RuntimeError("selected face does not have an embedding")

        return embedding

    def embed_aligned(self, crop_112: ImageInput) -> Embedding:
        """Embed one pre-aligned crop without detection.

        Args:
            crop_112: Pre-aligned 112×112 RGB image source.

        Returns:
            A normalized face embedding.
        """
        return self.embedder.embed(load_image(crop_112))

    def compare(
        self,
        a: ImageInput,
        b: ImageInput,
        *,
        metric: str | None = None,
        threshold: float | None = None,
        select: Selector = "largest",
    ) -> ComparisonResult:
        """Compare one selected face from each image.

        Args:
            a: First image source.
            b: Second image source.
            metric: Optional decision-metric override.
            threshold: Optional cosine-threshold override.
            select: Policy for choosing a face in each image.

        Returns:
            Similarity metrics and the threshold decision.
        """
        _cosine_metric(self.metric if metric is None else metric)
        boundary = self.threshold if threshold is None else _threshold(threshold)
        left = self.embed(a, select=select)
        right = self.embed(b, select=select)
        cosine_score = float(np.clip(dot(left, right), -1.0, 1.0))
        distance = euclidean(left, right)
        return ComparisonResult(
            cosine=cosine_score,
            euclidean=distance,
            percentage=to_percentage(cosine_score, threshold=boundary),
            is_match=cosine_score >= boundary,
            threshold=boundary,
        )

    def verify(self, a: ImageInput, b: ImageInput, **kwargs: Any) -> bool:
        """Return whether two selected faces meet the threshold.

        Args:
            a: First image source.
            b: Second image source.
            **kwargs: Options forwarded to :meth:`compare`.

        Returns:
            ``True`` when the selected faces match.
        """
        return self.compare(a, b, **kwargs).is_match

    def find(
        self,
        query: ImageCollection,
        gallery: ImageCollection,
        *,
        select: Selector = "largest",
        top_k: int = 1,
        threshold: float | None = None,
    ) -> list[Match]:
        """Find one selected query face in a gallery.

        Args:
            query: Query image source or collection.
            gallery: Gallery image source or collection.
            select: Policy for choosing the query face.
            top_k: Maximum number of matches to return.
            threshold: Optional cosine-threshold override.

        Returns:
            Matching gallery faces ordered by descending similarity.
        """
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        boundary = self.threshold if threshold is None else _threshold(threshold)
        query_face = self._select(self._collect(query), select)
        candidate_faces = self._collect(gallery)
        matrix = self._similarity_matrix([query_face], candidate_faces)
        ranked_indices = sorted(
            range(len(candidate_faces)),
            key=lambda index: (
                _descending_score(float(matrix[0, index])),
                _face_key(candidate_faces[index]),
            ),
        )
        return [
            self._match(query_face, candidate_faces[index], float(matrix[0, index]), boundary)
            for index in ranked_indices
            if matrix[0, index] >= boundary
        ][:top_k]

    def match(
        self,
        source_a: ImageCollection,
        source_b: ImageCollection,
        *,
        assignment: Assignment = "greedy",
        threshold: float | None = None,
    ) -> MatchResult:
        """Compare all faces in two collections and assign pairs.

        Args:
            source_a: Query image source or collection.
            source_b: Candidate image source or collection.
            assignment: Pair-selection strategy.
            threshold: Optional cosine-threshold override.

        Returns:
            Similarity matrix and selected face pairs.
        """
        if assignment not in {"greedy", "hungarian", "all"}:
            raise ValueError(f"unknown assignment method: {assignment!r}")

        boundary = self.threshold if threshold is None else _threshold(threshold)
        query_faces = self._collect(source_a)
        candidate_faces = self._collect(source_b)
        matrix = self._similarity_matrix(query_faces, candidate_faces)

        if assignment == "hungarian":
            indices = self._hungarian_indices(matrix, boundary)
        else:
            indices = self._ranked_indices(matrix, query_faces, candidate_faces, boundary)
            if assignment == "greedy":
                used_queries: set[int] = set()
                used_candidates: set[int] = set()
                selected: list[tuple[int, int]] = []

                for query_index, candidate_index in indices:
                    if query_index in used_queries or candidate_index in used_candidates:
                        continue

                    selected.append((query_index, candidate_index))
                    used_queries.add(query_index)
                    used_candidates.add(candidate_index)

                indices = selected

        pairs = [
            self._match(
                query_faces[query_index],
                candidate_faces[candidate_index],
                float(matrix[query_index, candidate_index]),
                boundary,
            )
            for query_index, candidate_index in indices
        ]
        return MatchResult(matrix, query_faces, candidate_faces, pairs, boundary)

    def group(
        self,
        images: ImageCollection,
        *,
        threshold: float | None = None,
        one_per_image: bool = True,
    ) -> list[list[Face]]:
        """Group faces into complete-linkage identity clusters.

        Args:
            images: Image source or collection to group.
            threshold: Optional cosine-threshold override.
            one_per_image: Prevent two faces from one image entering the same cluster.

        Returns:
            Stable clusters of mutually similar faces.
        """
        if not isinstance(one_per_image, bool):
            raise TypeError("one_per_image must be a boolean")

        boundary = self.threshold if threshold is None else _threshold(threshold)
        faces = sorted(self._collect(images), key=_face_key)
        if not faces:
            return []

        matrix = self._similarity_matrix(faces, faces)
        clusters = [[index] for index in range(len(faces))]
        memberships = list(range(len(faces)))

        # Strongest pairs are considered first; complete linkage is checked below.
        candidate_pairs = sorted(
            (
                (left, right)
                for left in range(len(faces))
                for right in range(left + 1, len(faces))
                if matrix[left, right] >= boundary
            ),
            key=lambda pair: (
                _descending_score(float(matrix[pair[0], pair[1]])),
                _face_key(faces[pair[0]]),
                _face_key(faces[pair[1]]),
            ),
        )

        for left, right in candidate_pairs:
            left_cluster = memberships[left]
            right_cluster = memberships[right]
            if left_cluster == right_cluster:
                continue

            left_members = clusters[left_cluster]
            right_members = clusters[right_cluster]
            if one_per_image and {faces[index].image_index for index in left_members} & {
                faces[index].image_index for index in right_members
            }:
                continue

            cross_scores = matrix[np.ix_(left_members, right_members)]
            if np.any(cross_scores < boundary):
                continue

            merged = sorted(left_members + right_members)
            keep = min(left_cluster, right_cluster)
            discard = max(left_cluster, right_cluster)
            clusters[keep] = merged
            clusters[discard] = []

            for index in merged:
                memberships[index] = keep

        grouped = [[faces[index] for index in cluster] for cluster in clusters if cluster]
        return sorted(grouped, key=lambda cluster: _face_key(cluster[0]))

    def _collect(self, source: ImageCollection) -> list[Face]:
        """Analyze image sources and assign collection indices.

        Args:
            source: Image source or collection.

        Returns:
            Faces labeled with their image and face positions.
        """
        collected: list[Face] = []
        for image_index, item in enumerate(_sources(source)):
            for face in self.analyze(item):
                collected.append(
                    Face(
                        bbox=face.bbox,
                        kps=face.kps,
                        aligned=face.aligned,
                        embedding=face.embedding,
                        image_index=image_index,
                        face_index=face.face_index,
                    )
                )
        return collected

    @staticmethod
    def _similarity_matrix(
        query_faces: list[Face],
        candidate_faces: list[Face],
    ) -> npt.NDArray[np.float64]:
        """Compute pairwise cosine similarity for two face lists.

        Args:
            query_faces: Faces forming matrix rows.
            candidate_faces: Faces forming matrix columns.

        Returns:
            Pairwise cosine-similarity matrix.
        """
        if not query_faces or not candidate_faces:
            return np.empty((len(query_faces), len(candidate_faces)), dtype=np.float64)

        def stack(faces: list[Face]) -> npt.NDArray[np.float32]:
            """Stack face embeddings into a float32 matrix.

            Args:
                faces: Faces whose embeddings should be stacked.

            Returns:
                A matrix containing one embedding per row.
            """
            embeddings = [face.embedding for face in faces]
            if any(embedding is None for embedding in embeddings):
                raise RuntimeError("analyzed face does not have an embedding")

            return np.stack([cast(Embedding, embedding).vector for embedding in embeddings])

        return pairwise_cosine(stack(query_faces), stack(candidate_faces))

    @staticmethod
    def _ranked_indices(
        matrix: npt.NDArray[np.float64],
        query_faces: list[Face],
        candidate_faces: list[Face],
        threshold: float,
    ) -> list[tuple[int, int]]:
        """Rank all above-threshold matrix coordinates.

        Args:
            matrix: Pairwise similarity matrix.
            query_faces: Faces corresponding to matrix rows.
            candidate_faces: Faces corresponding to matrix columns.
            threshold: Minimum accepted similarity.

        Returns:
            Matrix coordinates ordered by score and stable face keys.
        """
        return sorted(
            (
                (int(query_index), int(candidate_index))
                for query_index, candidate_index in np.argwhere(matrix >= threshold)
            ),
            key=lambda pair: (
                _descending_score(float(matrix[pair[0], pair[1]])),
                _face_key(query_faces[pair[0]]),
                _face_key(candidate_faces[pair[1]]),
            ),
        )

    @staticmethod
    def _hungarian_indices(
        matrix: npt.NDArray[np.float64],
        threshold: float,
    ) -> list[tuple[int, int]]:
        """Select optimal one-to-one matches with Hungarian assignment.

        Args:
            matrix: Pairwise similarity matrix.
            threshold: Minimum accepted similarity.

        Returns:
            Selected row and column index pairs.
        """
        query_count, candidate_count = matrix.shape
        if query_count == 0 or candidate_count == 0:
            return []

        try:
            optimize = importlib.import_module("scipy.optimize")
        except ModuleNotFoundError as error:
            if error.name == "scipy":
                raise ImportError(
                    'Hungarian assignment requires `pip install "lvface[hungarian]"`'
                ) from error
            raise

        linear_sum_assignment = optimize.linear_sum_assignment

        # Extra rows and columns let either side remain unmatched.
        maximum = float(np.max(matrix))
        minimum = float(np.min(matrix))
        unmatched_cost = maximum - minimum + 1.0
        invalid_cost = 2.0 * unmatched_cost + 1.0
        size = query_count + candidate_count
        costs = np.full((size, size), invalid_cost, dtype=np.float64)
        real_costs = maximum - matrix
        costs[:query_count, :candidate_count] = np.where(
            matrix >= threshold,
            real_costs,
            invalid_cost,
        )
        costs[np.arange(query_count), candidate_count + np.arange(query_count)] = unmatched_cost
        costs[query_count + np.arange(candidate_count), np.arange(candidate_count)] = unmatched_cost
        costs[query_count:, candidate_count:] = 0.0
        rows, columns = linear_sum_assignment(costs)
        return [
            (int(row), int(column))
            for row, column in zip(rows, columns, strict=True)
            if row < query_count and column < candidate_count and matrix[row, column] >= threshold
        ]

    @staticmethod
    def _match(query: Face, candidate: Face, score: float, threshold: float) -> Match:
        """Build a match record for a face pair.

        Args:
            query: Query face.
            candidate: Candidate face.
            score: Cosine similarity.
            threshold: Decision threshold.

        Returns:
            A populated match record.
        """
        return Match(
            query=query,
            candidate=candidate,
            score=score,
            percentage=to_percentage(score, threshold=threshold),
            is_match=score >= threshold,
        )

    @staticmethod
    def _select(faces: list[Face], select: Selector) -> Face:
        """Select one face according to an explicit policy.

        Args:
            faces: Available faces.
            select: Selection policy.

        Returns:
            The selected face.
        """
        if select not in {"largest", "highest_score", "error"}:
            raise ValueError(f"unknown face selector: {select!r}")

        if not faces:
            raise NoFaceError("no face was detected in the image")

        if len(faces) == 1:
            return faces[0]

        if select == "error":
            raise ValueError(f"expected exactly one face, found {len(faces)}")

        warnings.warn(
            f"found {len(faces)} faces; selecting the {select.replace('_', ' ')} face",
            RuntimeWarning,
            stacklevel=3,
        )
        if select == "largest":
            if any(face.bbox is None for face in faces):
                raise ValueError("largest selection requires bounding boxes")

            def area(face: Face) -> float:
                """Return the nonnegative bounding-box area.

                Args:
                    face: Face with a bounding box.

                Returns:
                    Bounding-box area.
                """
                width, height = cast(BBox, face.bbox).wh
                return float(max(0.0, width) * max(0.0, height))

            return max(faces, key=area)

        if any(face.bbox is None for face in faces):
            raise ValueError("highest_score selection requires detection scores")

        def score(face: Face) -> float:
            """Return the face detection score.

            Args:
                face: Face with a bounding box.

            Returns:
                Detection confidence.
            """
            return float(cast(BBox, face.bbox).score)

        return max(faces, key=score)
