"""
Bundles a graph's edges on gpu.
Faster than hammer bundle on larger graphs (>1000) thanks to cufft and parallel computations.

    Input:  cudf (or pandas) of normalized node positions and edge adjacency matrix
    Output: cudf of lines to be datashaded

Issues + possible improvements:
- Can easilly run out of gpu memory when using a large edge count + high sample rate. Segment batching?
- Very difficult to predict what parameters will result in a good looking graph, especially with larger graphs. Graph shape can greatly affect this.
- Weighted edges?
- Edge segments should go straight from cupy array to cudf instead of through numpy. Easy fix, don't have time.
- More performance.

Use of fft for fast gradient generation modeled after 'FFTEB: Edge Bundling of Huge Graphs by the Fast Fourier Transform' by A. Lhuillier, C. Hurter, and A. Telea.
    http://recherche.enac.fr/~hurter/FFTEB_WEB/FFTEB
"""

import param

import numpy  as np
import cupy   as cp
import pandas as pd
import cudf


# Get edges from input dataframes into cupy array
def read_dataframes(nodes, edges):

    # Input dataframes should be cudf for performance
    # pandas -> cudf -> cupy is faster than pandas -> cupy
    if (isinstance(nodes, pd.DataFrame)):
        nodes = cudf.from_pandas(nodes)

    if (isinstance(edges, pd.DataFrame)):
        edges = cudf.from_pandas(edges)

    # Construct edge segment array from dataframe
    df = cudf.merge(edges, nodes, left_on=["source"], right_index=True)
    df = df.rename(columns={'x': 'src_x', 'y': 'src_y'})

    df = cudf.merge(df, nodes, left_on=["target"], right_index=True)
    df = df.rename(columns={'x': 'dst_x', 'y': 'dst_y'})

    df = df[['src_x', 'src_y', 'dst_x', 'dst_y']]

    return df.to_cupy().flatten()


# Generate the fft of an offset gaussian kernel
def generate_kernel(kernel_size, width):
    
    x = cp.exp(-(cp.arange(width)-width/2)**2 / (2*kernel_size**2))
    y = cp.exp(-(cp.arange(width)-width/2)**2 / (2*kernel_size**2))
    kernel = cp.outer(x, y)
    
    kernel = cp.reshape(kernel, (width, width))    
    kernel = cp.roll(kernel, width//2, axis=(0, 1))
    
    fft_kernel = cp.fft.fft2(kernel)
    fft_kernel = fft_kernel.flatten()
    
    return fft_kernel


# Interpolate fixed number of edge segments for each edge quickly
def initial_sample(edge_segments, sample_distance):

    n = int(1 / sample_distance)

    x_positions = edge_segments[0::2]
    y_positions = edge_segments[1::2]

    # Get start and end points
    new_x_positions = cp.empty(x_positions.size + (x_positions.size // 2 * n))
    new_x_positions[  0::n+2] = x_positions[0::2]
    new_x_positions[n+1::n+2] = x_positions[1::2]

    new_y_positions = cp.empty(y_positions.size + (y_positions.size // 2 * n))
    new_y_positions[  0::n+2] = y_positions[0::2]
    new_y_positions[n+1::n+2] = y_positions[1::2]

    # Interpolate new points
    for i in range(1, n+1):
        new_x_positions[i::n+2] = x_positions[0::2] + (x_positions[1::2] - x_positions[0::2]) * i / (n+1)
        new_y_positions[i::n+2] = y_positions[0::2] + (y_positions[1::2] - y_positions[0::2]) * i / (n+1)

    # Merge x and y
    edge_segments = cp.empty(new_x_positions.size + new_y_positions.size)
    edge_segments[0::2] = new_x_positions
    edge_segments[1::2] = new_y_positions

    # Store beginning index of each edge segment
    offsets_buffer = cp.arange(0, edge_segments.size//2, n+2)

    return edge_segments, offsets_buffer


# Interpolate new edge segments when points are too far
def split_segments(edge_segments, offsets_buffer, split_distance):

    split     = cp.empty(edge_segments.size*2)
    distance  = cp.empty(edge_segments.size)
    bool_mask = cp.full(edge_segments.size, True)

    # Get distance between vertices
    distance = (edge_segments - cp.roll(edge_segments, -2))**2
    distance = cp.sqrt(distance[0::2] + distance[1::2])

    # Get existing vertices
    split[0::4] = edge_segments[0::2]
    split[1::4] = edge_segments[1::2]

    # Interpolate new vertices
    split[2::4] = (edge_segments[0::2] + cp.roll(edge_segments, -2)[0::2])/2
    split[3::4] = (edge_segments[1::2] + cp.roll(edge_segments, -2)[1::2])/2

    # Discard unneeded vertices
    bool_mask[offsets_buffer-1] = False                           # Ignore vertices between end points
    bool_mask[1::2] = bool_mask[1::2] & (distance>split_distance) # Ignore vertices where distance is not greater than split distance
    bool_mask = cp.repeat(bool_mask, 2)                           # Repeat each element twice for both x and y value

    edge_segments = split[bool_mask]

    # Update offset buffer
    positions = cp.zeros(bool_mask.size)
    positions[0::2] = bool_mask[0::2].astype(int)
    positions[offsets_buffer*2-1] = 2
    positions = positions[cp.nonzero(positions)]
    positions = cp.nonzero(positions-1)[0]
    offsets   = cp.arange(positions.size)            
    offsets_buffer = cp.r_[0, (positions-offsets)*2]

    return edge_segments, offsets_buffer


# Drop edge segments when points are too close
def merge_segments(edge_segments, offsets_buffer, merge_distance):

    distance  = cp.empty(edge_segments.size)
    bool_mask = cp.full(edge_segments.size//2, False)

    # Get distance between vertices
    distance = (edge_segments - cp.roll(edge_segments, -2))**2
    distance = cp.sqrt(edge_segments[0::2] + edge_segments[1::2])

    # Add needed vertices
    bool_mask[cp.concatenate((offsets_buffer//2, offsets_buffer//2-1))] = True 
    bool_mask = bool_mask | (distance>merge_distance)
    bool_mask = cp.repeat(bool_mask, 2)

    edge_segments = edge_segments[bool_mask]

    # Update offset buffer
    positions = cp.zeros(bool_mask.size)
    positions[0::2] = bool_mask[0::2].astype(int)
    positions[offsets_buffer-1] = 2
    positions = positions[cp.nonzero(positions)]
    positions = cp.nonzero(positions-1)[0]
    offsets   = cp.arange(positions.size)          
    offsets_buffer = cp.r_[0, (positions-offsets)*2]

    return edge_segments, offsets_buffer


# Move edge segments along gradient
def advect_edge_segments(edge_segments, segment_positions, gradient, width, move_distance):

    # Apply gradient
    dx = gradient[segment_positions+1]     - gradient[segment_positions-1]
    dy = gradient[segment_positions+width] - gradient[segment_positions-width]

    # Normalize movement
    norm = cp.sqrt(dx * dx + dy * dy) / (move_distance / width)
    norm[norm==0] = 0.0001
    dx /= norm; dy /= norm
    
    # Move edge segments
    edge_segments[0::2] += dx
    edge_segments[1::2] += dy
    edge_segments = cp.clip(edge_segments, 0, 1)

    return edge_segments


# Smooth edges
def smooth(edge_segments, offsets_buffer, end_segments, iterations):
    
    for i in range(iterations):

        # Smooth
        edge_segments = (cp.roll(edge_segments, -2) + edge_segments + cp.roll(edge_segments, 2)) / 3

        # Reset end edge segments
        for j in range(4):
            edge_segments[offsets_buffer[:-1]+(j-2)] = end_segments[j::4]

    return(edge_segments)


class fft_bundle(param.ParameterizedFunction):


    iterations = param.Integer(default=40,bounds=(1, None),doc="""
        Number of passes for the edge bundling.""")

    accuracy = param.Integer(default=400,bounds=(1, None),precedence=-0.5,doc="""
        Resolution of possible edge positions.""")

    move_distance = param.Number(default=2,bounds=(0.0, None),doc="""
        How far an edge will be moved along the gradient.""")

    kernel_size = param.Number(default=0.05,bounds=(0.0, 1.0),doc="""
        Initial size of the gaussian kernel.""")

    kernel_decay = param.Number(default=0.95,bounds=(0.0, 1.0),doc="""
        Rate of gaussian kernel decay.""")

    resample_frequency = param.Integer(default=3,bounds=(1, None),doc="""
        How often edges will be resampled.""")

    resample_distance = param.Number(default=0.02,bounds=(0.0, 1.0),doc="""
        How far points on an edge must be apart before new point is inserted.""")

    final_smooth = param.Integer(default=1,bounds=(0, None),doc="""
        Number of smoothing operations that will be performed on final graph.""")



    def __call__(self, nodes, edges, **params):

        p = param.ParamOverrides(self, params)
        width = p.accuracy


        # Convert input dataframe to array of edges
        edges = read_dataframes(nodes, edges)
        
        # Store original positions
        end_segments = edges

        # Perform initial interpolation
        edge_segments, offsets_buffer = initial_sample(edges, p.resample_distance)


        # Track when kernel size changes
        prev_kernel_size = 0

        for i in range(p.iterations):
            
            ### Generate gaussian kernel of decreasing size
            # Kernel must be even to ensure symmetry of kernel
            kernel_size = int(p.accuracy * p.kernel_size * p.kernel_decay**i / 2) * 2

            # End bundling early if kernel size becomes too low to effect graph
            if kernel_size < 2:
                break

            # Generate new kernel if needed
            if kernel_size != prev_kernel_size:
                fft_kernel = generate_kernel(kernel_size, width)
                prev_kernel_size = kernel_size


            ### Resample edge segments periodically
            if (i % p.resample_frequency == 0):

                # Interpolate new edge segments
                edge_segments, offsets_buffer = split_segments(edge_segments, offsets_buffer, p.resample_distance)

                # Remove clustered edge segments
                edge_segments, offsets_buffer = merge_segments(edge_segments, offsets_buffer, p.resample_distance/2)


            ### Generate and transform edge segment density map
            # Convert normalized x/y positions to 1d indicies
            segment_positions = (edge_segments * width + 0.1).astype(int)
            segment_positions = segment_positions[0::2].get() + (segment_positions[1::2].get() * width)

            density = cp.zeros((width * width))
            density[segment_positions] = 1

            fft_density = cp.fft.fft(density)


            ### Convolute gradient map
            fft_gradient = fft_kernel * fft_density

            # Discard imaginary portion and normalize
            gradient = (cp.fft.ifft(fft_gradient)).real
            gradient = (gradient - cp.min(gradient))/(cp.max(gradient) - cp.min(gradient))


            ### Move edge segments along gradient
            edge_segments = advect_edge_segments(edge_segments, segment_positions, gradient, width, p.move_distance)

            # Reset end segments
            for j in range(4):
                edge_segments[offsets_buffer[:-1]+(j-2)] = end_segments[j::4]


            ### Smooth edges, lightly
            edge_segments = smooth(edge_segments, offsets_buffer, end_segments, 1)
        

        # Smooth edges, potentially less lightly
        edge_segments = smooth(edge_segments, offsets_buffer, end_segments, p.final_smooth)


        # Seperate edges with NaNs and convert to cudf
        edge_segments = np.insert(edge_segments.get(), cp.concatenate((offsets_buffer[1::], offsets_buffer[1::])).get(), np.nan) # cupy has no insert? this should go straight to cudf, not numpy
        df = cudf.DataFrame({'x': edge_segments[0::2], 'y': edge_segments[1::2]})

        return df